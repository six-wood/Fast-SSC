import torch
import torch_scatter
import torch.nn as nn

from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmdet3d.utils import ConfigType, OptMultiConfig
from typing import Sequence, List, Optional, Tuple
from mmcv.cnn import build_norm_layer, build_activation_layer
import torch.utils.checkpoint as cp

from .resnet34 import ResNet34


@MODELS.register_module()
class VFE(nn.Module):
    """Voxel feature encoder used in segmentation task.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 6.
        feat_channels (list(int)): Channels of features in VFE.
        with_voxel_center (bool): Whether to use the distance
            to center of voxel for each points inside a voxel.
            Defaults to False.
        voxel_size (tuple[float]): Size of a single voxel (rho, phi, z).
            Defaults to None.
        grid_shape (tuple[float]): The grid shape of voxelization.
            Defaults to (480, 360, 32).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Defaults to (0, -3.14159265359, -4, 50, 3.14159265359, 2).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points
            inside a voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        with_pre_norm (bool): Whether to use the norm layer before
            input vfe layer.
        feat_compression (int, optional): The voxel feature compression
            channels, Defaults to None
        return_point_feats (bool): Whether to return the features
            of each points. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Sequence[int] = [],
        with_voxel_center: bool = False,
        with_distance: bool = False,
        voxel_size: Optional[Sequence[float]] = None,
        point_cloud_range: Sequence[float] = (0, -25.6, -2, 51.2, 25.6, 4.4),
        norm_cfg: dict = dict(type="BN1d"),
        mode: str = "max",
        with_pre_norm: bool = True,
        feat_compression: Optional[int] = None,
        return_point_feats: bool = False,
        act_cfg: ConfigType = dict(type=nn.Hardswish, inplace=True),
    ) -> None:
        super(VFE, self).__init__()
        assert mode in ["avg", "max"]
        assert len(feat_channels) > 0

        in_channels = in_channels + 3 if with_distance else in_channels
        in_channels = in_channels + 1 if with_voxel_center else in_channels

        self.in_channels = in_channels
        self._with_voxel_center = with_voxel_center
        self._with_distance = with_distance
        self.return_point_feats = return_point_feats

        self.point_cloud_range = point_cloud_range
        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)

        self.voxel_size = voxel_size
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_shape = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_shape = torch.round(grid_shape).long().tolist()
        self.grid_shape = grid_shape

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = self.voxel_size[0]
        self.vy = self.voxel_size[1]
        self.vz = self.voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        feat_channels = [self.in_channels] + list(feat_channels)
        self.pre_norm = build_norm_layer(norm_cfg, self.in_channels)[1] if with_pre_norm else None

        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                vfe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                vfe_layers.append(nn.Sequential(nn.Linear(in_filters, out_filters, bias=False), norm_layer, build_activation_layer(act_cfg)))
        self.vfe_layers = nn.ModuleList(vfe_layers)

        self.compression_layers = (
            nn.Sequential(nn.Linear(feat_channels[-1], feat_compression), build_activation_layer(act_cfg)) if feat_compression is not None else None
        )

    def forward(self, features: torch.Tensor, coors: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor]:
        """Forward functions.

        Args:
            features (Tensor): Features of voxels, shape is NxC.
            coors (Tensor): Coordinates of voxels, shape is  Nx(1+NDim).

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels additionally.
        """
        features_ls = [features]

        # Find distance of x, y, and z from voxel center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = (features[:, 0] - (coors[:, 1].type_as(features) * self.vx + self.x_offset)) / self.vx
            f_center[:, 1] = (features[:, 1] - (coors[:, 2].type_as(features) * self.vy + self.y_offset)) / self.vy
            f_center[:, 2] = (features[:, 2] - (coors[:, 3].type_as(features) * self.vz + self.z_offset)) / self.vz
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls[::-1], dim=-1)
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        for vfe in self.vfe_layers:
            features = vfe(features)

        voxel_coors, inv = torch.unique(coors, return_inverse=True, dim=0)
        voxel_feats = torch_scatter.scatter_max(features, inv, dim=0)[0]
        voxel_feats = self.compression_layers(voxel_feats) if self.compression_layers is not None else voxel_feats

        dense_voxel_fea = torch.zeros(
            (voxel_coors[-1, 0] + 1, voxel_feats.size(1), self.grid_shape[2], self.grid_shape[1], self.grid_shape[0]),
            device=voxel_feats.device,
            dtype=voxel_feats.dtype,
        )
        dense_voxel_fea[voxel_coors[:, 0], :, voxel_coors[:, 3], voxel_coors[:, 2], voxel_coors[:, 1]] = voxel_feats

        return dense_voxel_fea


@MODELS.register_module()
class BevBackbone(BaseModule):
    """ResNet34-based Bev Backbone for SSC

    Args:
        BaseModule (_type_): _description_
    """

    def __init__(
        self,
        in_channel: int = 128,
        proj_channel: int = 128,
        num_stages: int = 4,
        h_size: int = 32,
        stage_blocks: Sequence[int] = (3, 4, 6, 3),
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        encoder_out_channels: Sequence[int] = (128, 128, 128, 128),
        norm_cfg: ConfigType = dict(type="BN2d"),
        act_cfg: ConfigType = dict(type=nn.Hardswish, inplace=True),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(BevBackbone, self).__init__(init_cfg)

        self.resnet34 = ResNet34(
            in_channel=in_channel,
            proj_channel=proj_channel,
            num_stages=num_stages,
            h_size=h_size,
            stage_blocks=stage_blocks,
            out_channels=encoder_out_channels,
            strides=strides,
            dilations=dilations,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet34(x)


class PDB(nn.Module):
    """Parallel-Dilation Block for completion Networks.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: List[int],
        norm_cfg: ConfigType = dict(type=nn.InstanceNorm3d),
        act_cfg: ConfigType = dict(type=nn.Hardswish, inplace=True),
        with_cp: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        self.dilations = dilations
        self.aspp = nn.ModuleList()
        self.norm = nn.ModuleList()
        for dilation in self.dilations:
            self.aspp.append(nn.Conv3d(self.in_channels, self.out_channels, 3, stride=1, padding=dilation, dilation=dilation, bias=False))
            self.norm.append(build_norm_layer(norm_cfg, self.out_channels)[1])
        self.conv_out = nn.Conv3d(self.out_channels * len(self.dilations), self.out_channels, 1, stride=1, padding=0, dilation=1, bias=False)
        self.norm_out = build_norm_layer(norm_cfg, self.out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)
        self.with_cp = with_cp

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        def _inner_forward(x):
            y_list = []
            for i in range(0, len(self.dilations)):
                x_i = self.aspp[i](x)
                x_i = self.norm[i](x_i)
                x_i = self.act(x_i)
                y_list.append(x_i)

            y = torch.cat(y_list, dim=1)
            y = self.conv_out(y)
            y = self.norm_out(y)
            y = y + x
            return y

        if self.with_cp:
            y = cp.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            y = _inner_forward(x)
        y = self.act(y)

        return y


@MODELS.register_module()
class SparseBackbone(BaseModule):
    """MDB-based Sparse Backbone for SSC

    Args:
        BaseModule (_type_): _description_
    """

    def __init__(
        self,
        in_channels: int = 32,
        norm_cfg: ConfigType = dict(type=nn.InstanceNorm3d),
        act_cfg: ConfigType = dict(type=nn.Hardswish, inplace=True),
        init_cfg: OptMultiConfig = None,
        with_cp: bool = False,
    ):
        super().__init__(init_cfg)

        pdb1 = PDB(in_channels, in_channels, [1, 2, 3], norm_cfg=norm_cfg, act_cfg=act_cfg, with_cp=with_cp)
        pdb2 = PDB(in_channels, in_channels, [1, 2, 5], norm_cfg=norm_cfg, act_cfg=act_cfg, with_cp=with_cp)
        self.pdb3 = PDB(in_channels, in_channels, [1, 3, 5], norm_cfg=norm_cfg, act_cfg=act_cfg, with_cp=with_cp)
        self.pdb12 = nn.Sequential(pdb1, pdb2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pdb12 = self.pdb12(x)
        x_pdb3 = self.pdb3(x)

        return x_pdb12 + x_pdb3
