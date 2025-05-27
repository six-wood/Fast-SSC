from typing import Optional, Sequence, Tuple

import torch
from mmcv.cnn import ConvModule, build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor, nn
from torch.nn import functional as F
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig


class SAEA(nn.Module):
    """Space enhances attention

    Returns:
       tensor: The output tensor.
    """

    def __init__(self, h_size, channel: int = 128, k_size: int = 3) -> None:
        super().__init__()
        # z branch
        self.h_channel = channel // h_size
        self.h_size = h_size
        self.conv_compress = nn.Conv3d(self.h_channel, 1, kernel_size=(k_size, 1, 1), stride=(1, 1, 1), padding=(k_size // 2, 0, 0), bias=False)

        # yx branch
        self.yh_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.xw_pool = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

        # fuse
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, fea: Tensor) -> Tensor:
        # z branch
        b, _, yh, xw = fea.size()
        fea_3d = fea.view(b, self.h_channel, self.h_size, yh, xw)  # b, -1, z, y, x
        fea_z = self.conv_compress(fea_3d)  # b, 1, z, y, x
        fea_z = F.sigmoid(fea_z)  # b, 1, z, y, x
        fea_z = (fea_3d * fea_z).view(b, -1, yh, xw)  # b, c, y, x
        # yx branch
        fea_yh = self.yh_pool(fea)  # b, c, y, 1
        fea_xw = self.xw_pool(fea).permute(0, 1, 3, 2).contiguous()  # b, c, x, 1
        hw = self.conv1x1(torch.cat([fea_yh, fea_xw], dim=2))  # b, c, y+x, 1
        fea_yh, fea_xw = torch.split(hw, [yh, xw], dim=2)  # b, c, y, 1; b, c, x, 1
        fea_yx = fea * fea_yh.sigmoid() * fea_xw.permute(0, 1, 3, 2).contiguous().sigmoid()  # b, c, y, x

        # sample fuse
        w_yx = F.softmax(self.avg_pool(fea_yx), -1)
        w_z = F.softmax(self.avg_pool(fea_z), -1)
        fea = F.sigmoid(w_yx * fea_z + w_z * fea_yx) * fea

        # fea = fea_z + fea_yx

        return fea

class BasicBlock(BaseModule):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        h_size: int = 32,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type=nn.BatchNorm2d),
        act_cfg: ConfigType = dict(type=nn.Hardswish, inplace=True),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(BasicBlock, self).__init__(init_cfg)

        self.norm1_name, self.norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, self.norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.act = build_activation_layer(act_cfg)
        self.att = SAEA(h_size, planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.att(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)
        return out


class ResNet34(nn.Module):
    def __init__(
        self,
        in_channel: int = 32 * 32,
        proj_channel: int = 128,
        num_stages: int = 4,
        h_size: int = 32,
        stage_blocks: Sequence[int] = (3, 4, 6, 3),
        out_channels: Sequence[int] = (128, 128, 128, 128),
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type=nn.BatchNorm2d),
        act_cfg: ConfigType = dict(type=nn.Hardswish, inplace=True),
    ) -> None:
        super().__init__()

        assert len(stage_blocks) == len(out_channels) == len(strides) == len(dilations) == num_stages, (
            "The length of stage_blocks, out_channels, strides and " "dilations should be equal to num_stages"
        )
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.proj = ConvModule(in_channel, proj_channel, 1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        inplanes = proj_channel
        self.res_layers = nn.ModuleList()
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = out_channels[i]
            res_layer = self.make_res_layer(
                inplanes=inplanes,
                planes=planes,
                h_size=h_size // (2**i),
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            inplanes = planes
            self.res_layers.append(res_layer)

    def make_res_layer(
        self,
        inplanes: int,
        planes: int,
        h_size: int,
        num_blocks: int,
        stride: int,
        dilation: int,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type=nn.BatchNorm2d),
        act_cfg: ConfigType = dict(type=nn.Hardswish, inplace=True),
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                build_conv_layer(conv_cfg, inplanes, planes, kernel_size=1, stride=stride, bias=False), build_norm_layer(norm_cfg, planes)[1]
            )

        layers = []
        layers.append(
            BasicBlock(
                inplanes=inplanes,
                planes=planes,
                h_size=h_size,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        )
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    inplanes=inplanes,
                    planes=planes,
                    h_size=h_size,
                    stride=1,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.proj(x.flatten(1, 2))
        outs = [x]
        for res_layer in self.res_layers:
            x = res_layer(x)
            outs.append(x)

        return tuple(outs)
