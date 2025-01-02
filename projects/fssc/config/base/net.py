import torch.nn as nn
from projects.fssc.plugin.model.ssc_net import SscNet
from projects.fssc.plugin.model.head import SscHead, ScHead
from projects.fssc.plugin.model.backbone import BevBackbone, SparseBackbone, VFE
from projects.fssc.plugin.model.neck import CustomFPN

from projects.fssc.plugin.model.loss import OccLovaszLoss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor

from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

HSwish = dict(type=nn.Hardswish, inplace=True)

SyncNorm = dict(type=nn.SyncBatchNorm, affine=True, track_running_stats=True)
InsNorm3d = dict(type=nn.InstanceNorm3d, affine=True, track_running_stats=True)

model = dict(
    type=SscNet,
    data_preprocessor=dict(type=Det3DDataPreprocessor),
    voxel_layer=dict(
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        max_voxels=(-1, -1),
    ),
    pts_voxel_encoder=dict(
        type=VFE,
        feat_channels=[64, 128, 256, 512],
        in_channels=4,
        with_voxel_center=True,
        with_distance=True,
        feat_compression=voxel_channel,
        return_point_feats=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=SyncNorm,
        act_cfg=HSwish,
    ),
    sparse_backbone=dict(
        type=SparseBackbone,
        in_channels=voxel_channel,
        norm_cfg=InsNorm3d,
        act_cfg=HSwish,
        with_cp=False,
    ),
    bev_backbone=dict(
        type=BevBackbone,
        in_channel=voxel_channel * grid_shape[2],
        proj_channel=128,
        num_stages=4,
        h_size=grid_shape[2],
        stage_blocks=(3, 4, 6, 3),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        encoder_out_channels=(128, 128, 128, 128),
        norm_cfg=SyncNorm,
        act_cfg=HSwish,
    ),
    neck=dict(
        type=CustomFPN,
        in_channels=[128, 128, 128, 128, 128],
        out_channels=128,
        num_outs=1,
        start_level=0,
        out_ids=[0],
        norm_cfg=SyncNorm,
        act_cfg=HSwish,
    ),
    ssc_head=dict(
        type=SscHead,
        indice=0,
        in_channels=128,
        grid_shape=grid_shape,
        num_classes=num_classes,
        loss_ce=dict(type=CrossEntropyLoss, class_weight=semantickitti_class_weight, ignore_index=ignore_index, avg_non_ignore=True),
        loss_lovasz=dict(type=OccLovaszLoss, reduction="none", ignore_index=ignore_index),
    ),
    sc_head=dict(
        type=ScHead,
        num_classes=2,
        grid_shape=grid_shape,
        in_channels=voxel_channel,
        loss_ce=dict(type=CrossEntropyLoss, class_weight=geo_class_weight, ignore_index=ignore_index, avg_non_ignore=True),
    ),
)
