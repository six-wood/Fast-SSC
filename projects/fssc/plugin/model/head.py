import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from typing import List, Dict

import torch
import numpy as np
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType, OptMultiConfig
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class SscHead(BaseModule):
    """
    Head for BEV Convolutional Networks.
    """

    def __init__(
        self,
        indice=0,
        in_channels: int = 128,
        num_classes: int = 20,
        loss_ce: ConfigType = None,
        loss_lovasz: ConfigType = None,
        grid_shape: List[int] = [256, 256, 32],
        init_cfg: OptMultiConfig = None,
        train_cfg: OptMultiConfig = None,
        test_cfg: OptMultiConfig = None,
        **kwargs,
    ):
        super(SscHead, self).__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.indice = indice
        self.num_classes = num_classes
        self.grid_shape = grid_shape

        self.outc = nn.Conv2d(in_channels, grid_shape[2] * num_classes, kernel_size=1, stride=1, padding=0)

        self.loss_ce = MODELS.build(loss_ce) if loss_ce is not None else None
        self.loss_lovasz = MODELS.build(loss_lovasz) if loss_lovasz is not None else None

    def forward(self, x: List[Tensor]) -> Tensor:
        ssc = self.outc(x[self.indice]).view(-1, self.num_classes, self.grid_shape[2], self.grid_shape[1], self.grid_shape[0]).permute(0, 1, 4, 3, 2)
        return ssc

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concat voxel-wise Groud Truth."""
        gt_semantic_segs = np.stack([data_sample.metainfo["voxel_label"] for data_sample in batch_data_samples], axis=0)
        return torch.from_numpy(gt_semantic_segs).long()

    def loss_by_feat(self, geo_logits: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            seg_logit (Tensor): Predicted per-point segmentation logits of
                shape [B, num_classes, N].
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        ssc_label = self._stack_batch_gt(batch_data_samples).to(geo_logits.device)
        losses = dict()
        losses["loss_ce"] = self.loss_ce(geo_logits, ssc_label) if self.loss_ce is not None else None
        losses["loss_lovasz"] = self.loss_lovasz(geo_logits, ssc_label) if self.loss_lovasz is not None else None
        # remove None in the dict
        losses = {k: v for k, v in losses.items() if v is not None}
        return losses

    def loss(self, x, batch_data_samples: SampleList, train_cfg: ConfigType = None) -> Dict[str, Tensor]:
        """Forward function for training.

        Args:
            inputs (dict): Feature dict from backbone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            train_cfg (dict or :obj:`ConfigDict`): The training config.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        ssc_logits = self.forward(x)

        losses = self.loss_by_feat(ssc_logits, batch_data_samples)
        return losses

    def predict(self, x) -> List[Tensor]:
        """Forward function for testing.

        Args:
            inputs (Tensor): Features from backone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples.

        Returns:
            List[Tensor]: The segmentation prediction mask of each batch.
        """

        ssc_pred = self.forward(x).argmax(dim=1)
        return ssc_pred


@MODELS.register_module()
class ScHead(BaseModule):
    """
    Head for BEV Convolutional Networks.
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_classes: int = 2,
        loss_ce: ConfigType = None,
        grid_shape: List[int] = [256, 256, 32],
        init_cfg: OptMultiConfig = None,
        train_cfg: OptMultiConfig = None,
        test_cfg: OptMultiConfig = None,
        **kwargs,
    ):
        super(ScHead, self).__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = num_classes
        self.grid_shape = grid_shape

        self.outc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm3d(in_channels),
            nn.Hardswish(),
            nn.Conv3d(in_channels, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.loss_ce = MODELS.build(loss_ce) if loss_ce is not None else None

    def forward(self, x: List[Tensor]) -> Tensor:
        sc = self.outc(x).permute(0, 1, 4, 3, 2)
        return sc

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concat voxel-wise Groud Truth."""
        gt_semantic_segs = np.stack([data_sample.metainfo["voxel_label"] for data_sample in batch_data_samples], axis=0)
        gt_semantic_segs = np.where((gt_semantic_segs > 0) & (gt_semantic_segs < 255), 1, gt_semantic_segs)
        return torch.from_numpy(gt_semantic_segs).long()

    def loss_by_feat(self, geo_logits: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            seg_logit (Tensor): Predicted per-point segmentation logits of
                shape [B, num_classes, N].
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        sc_label = self._stack_batch_gt(batch_data_samples).to(geo_logits.device)
        losses = dict()
        losses["loss_ce"] = self.loss_ce(geo_logits, sc_label) if self.loss_ce is not None else None
        # remove None in the dict
        losses = {k: v for k, v in losses.items() if v is not None}
        return losses

    def loss(self, x, batch_data_samples: SampleList, train_cfg: ConfigType = None) -> Dict[str, Tensor]:
        """Forward function for training.

        Args:
            inputs (dict): Feature dict from backbone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            train_cfg (dict or :obj:`ConfigDict`): The training config.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        sc_logits = self.forward(x)

        losses = self.loss_by_feat(sc_logits, batch_data_samples)
        return losses

    def predict(self, x) -> List[Tensor]:
        """Forward function for testing.

        Args:
            inputs (Tensor): Features from backone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples.

        Returns:
            List[Tensor]: The segmentation prediction mask of each batch.
        """

        sc_pred = self.forward(x).argmax(dim=1)
        return sc_pred
