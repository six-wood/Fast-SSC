import numpy as np
from mmdet3d.datasets.semantickitti_dataset import SemanticKittiDataset
from typing import Callable, List, Optional, Union
from os import path as osp
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class SemanticKittiSC(SemanticKittiDataset):
    METAINFO = {
        "classes": (
            "free",
            "car",
            "bicycle",
            "motorcycle",
            "truck",
            "other-vehicle",
            "person",
            "bicyclist",
            "motorcyclist",
            "road",
            "parking",
            "sidewalk",
            "other-ground",
            "building",
            "fence",
            "vegetation",
            "trunk",
            "terrian",
            "pole",
            "traffic-sign",
            "occupied",
        ),
        "palette": [
            [0, 0, 0],
            [100, 150, 245],
            [100, 230, 245],
            [30, 60, 150],
            [80, 30, 180],
            [100, 80, 250],
            [255, 30, 30],
            [255, 40, 200],
            [150, 30, 90],
            [255, 0, 255],
            [255, 150, 255],
            [75, 0, 75],
            [175, 0, 75],
            [255, 200, 0],
            [255, 120, 50],
            [0, 175, 0],
            [135, 60, 0],
            [150, 240, 80],
            [255, 240, 150],
            [255, 0, 0],
        ],
        "seg_valid_class_ids": tuple(range(21)),
        "seg_all_class_ids": tuple(range(21)),
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        ann_file: str = "",
        metainfo: Optional[dict] = None,
        data_prefix: dict = dict(pts="", img="", voxel_label=""),
        pipeline: List[Union[dict, Callable]] = [],
        modality: dict = dict(use_lidar=True, use_camera=False),
        ignore_index: Optional[int] = None,
        scene_idxs: Optional[Union[str, np.ndarray]] = None,
        test_mode: bool = False,
        voxel_size=[0.2, 0.2, 0.2],
        **kwargs,
    ) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs,
        )

        self.voxel_size = voxel_size
        self.img_W = 1220
        self.img_H = 370

    def parse_data_info(self, info: dict) -> dict:
        if self.modality["use_lidar"]:
            info["lidar_points"]["lidar_path"] = info["lidar_path"] = osp.join(self.data_root, info["lidar_points"]["lidar_path"])
            info["pts_semantic_mask_path"] = osp.join(self.data_root, info["pts_semantic_mask_path"])

            info["voxel_label_path"] = osp.join(self.data_root, info["voxel_label_path"])

            info["num_pts_feats"] = info["lidar_points"]["num_pts_feats"]
            info["voxel_size"] = info["lidar_points"]["voxel_size"]

        info["seg_label_mapping"] = self.seg_label_mapping
        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode and self.load_eval_anns:
            info["eval_ann_info"] = dict()

        return info
