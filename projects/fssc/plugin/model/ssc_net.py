from typing import Dict, Union, List
import os
import time
import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from mmdet3d.models import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.models.utils import add_prefix


@MODELS.register_module()
class SscNet(MVXTwoStageDetector):
    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        voxel_layer: ConfigType = None,
        pts_voxel_encoder: ConfigType = None,
        sparse_backbone: ConfigType = None,
        bev_backbone: ConfigType = None,
        neck: ConfigType = None,
        ssc_head: ConfigType = None,
        sc_head: ConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(SscNet, self).__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        self.voxel_layer = voxel_layer

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder) if pts_voxel_encoder is not None else None
        self.sparse_backbone = MODELS.build(sparse_backbone) if sparse_backbone is not None else None
        self.bev_backbone = MODELS.build(bev_backbone) if bev_backbone is not None else None
        self.neck = MODELS.build(neck) if neck is not None else None
        self.ssc_head = MODELS.build(ssc_head) if ssc_head is not None else None
        self.sc_head = MODELS.build(sc_head) if sc_head is not None else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.test_cfg is not None:
            self.save_path = self.test_cfg.get("save_path", None)
            os.makedirs(self.save_path, exist_ok=True)

            label_map_inv = self.test_cfg.get("labels_map_inv", None)
            maxkey = max(label_map_inv.keys())

            self.remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
            self.remap_lut[list(label_map_inv.keys())] = list(label_map_inv.values())

    @torch.no_grad()
    def voxelize(self, points: List[Tensor]) -> Dict[str, Tensor]:
        """Voxelize points to voxels.

        Args:
            points (List[Tensor]): Points of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            dict: Voxel dict.
        """
        voxel_dict = dict()
        coors = []
        voxels = []
        voxel_size = torch.tensor(self.voxel_layer["voxel_size"], device=points[0].device)
        point_cloud_range = torch.tensor(self.voxel_layer["point_cloud_range"], device=points[0].device)
        point_min, point_max = point_cloud_range[:3], point_cloud_range[3:]
        gird_size = torch.round((point_max - point_min) / voxel_size).int()

        for i, res in enumerate(points):
            res_coors = torch.floor((res[:, :3] - point_min) / voxel_size).int()
            voxels.append(res)
            coors.append(F.pad(res_coors, (1, 0), mode="constant", value=i))

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)

        mask = torch.all(coors[:, 1:] >= 0, dim=1) & torch.all(coors[:, 1:] < gird_size, dim=1)
        voxel_dict["voxels"] = voxels[mask]
        voxel_dict["coors"] = coors[mask]

        return voxel_dict

    def extract_pts_feat(self, voxel_dict: Dict[str, Tensor]) -> Tensor:
        """Extract features from point cloud.

        Args:
            voxel_dict (Dict[str, Tensor]): Voxel dict which includes 'voxels'

        Returns:
            Tensor: _description_
        """
        x = self.pts_voxel_encoder(voxel_dict["voxels"], voxel_dict["coors"])
        x = self.sparse_backbone(x)
        return x

    def extract_bev_feat(self, x: Tensor) -> Tensor:
        """Extract features from BEV images.

        Args:
            x (Tensor): BEV images with shape (B, C, H, W).

        Returns:
            Tensor: Extracted features from BEV images.
        """
        bev_feature = self.bev_backbone(x)  # b z y x

        return bev_feature

    def extract_feat(self, batch_inputs_dict):
        voxel_dict = self.voxelize(batch_inputs_dict["points"])

        geo_fea = self.extract_pts_feat(voxel_dict)
        bev_fea = self.extract_bev_feat(geo_fea)
        bev_fea = self.neck(bev_fea)

        fea = {"bev_fea": bev_fea, "geo_fea": geo_fea}
        return fea

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        # extract features using backbone
        fea = self.extract_feat(batch_inputs_dict)

        losses = dict()
        loss_ssc = self.ssc_head.loss(fea["bev_fea"], batch_data_samples) if self.ssc_head is not None else dict()
        loss_sc = self.sc_head.loss(fea["geo_fea"], batch_data_samples) if self.sc_head is not None else dict()
        losses.update(add_prefix(loss_ssc, "ssc")) if loss_ssc else None
        losses.update(add_prefix(loss_sc, "sc")) if loss_sc else None

        return losses

    def val_step(self, data):
        return self.test_step(data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        batch_inputs_dict = data["inputs"]
        batch_data_samples = data["data_samples"]

        time_start = time.time()
        fea = self.extract_feat(batch_inputs_dict)
        ssc_pred = self.ssc_head.predict(fea["bev_fea"]) if self.ssc_head is not None else None
        sc_pred = self.sc_head.predict(fea["geo_fea"]) if self.sc_head is not None else None
        time_end = time.time()

        time_cost = time_end - time_start

        if self.test_cfg is not None:
            for ssc_result, sc_result, batch_data in zip(ssc_pred, sc_pred, batch_data_samples):
                batch_data.set_data({"time_cost": time_cost})
                ssc_result = ssc_result.cpu().numpy()
                ssc_result = ssc_result.reshape(-1).astype(np.uint16)
                ssc_result = self.remap_lut[ssc_result].astype(np.uint16)
                lidar_path = batch_data.lidar_path
                name = lidar_path.split("/")[-1].split(".")[0]
                seq = lidar_path.split("/")[-3]
                save_path = os.path.join(self.save_path, "ssc_cache", "sequences", seq, "predictions", name + ".label")
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                ssc_result.tofile(save_path)

                sc_result = sc_result.cpu().numpy()
                sc_result = sc_result.reshape(-1).astype(np.uint16)
                save_path = os.path.join(self.save_path, "sc_cache", "sequences", seq, "predictions", name + ".label")
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                sc_result.tofile(save_path)
                
            if self.test_cfg.get("val", True):
                batch_data_samples = self.postprocess_result(ssc_pred, time_cost, batch_data_samples)

        elif self.test_cfg is None:
            batch_data_samples = self.postprocess_result(ssc_pred, time_cost, batch_data_samples)

        return batch_data_samples

    def postprocess_result(self, ssc_labels: Tensor, time_cost: float, batch_data_samples: SampleList) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            coors: b z y x

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        ssc_true = torch.from_numpy(np.stack([data_sample.metainfo["voxel_label"] for data_sample in batch_data_samples], axis=0)).to(ssc_labels.device)
        ssc_pred = ssc_labels.clone()

        for i, batch_data in enumerate(batch_data_samples):
            batch_data.set_data({"y_pred": ssc_pred[i]})
            batch_data.set_data({"y_true": ssc_true[i]})
            batch_data.set_data({"time_cost": time_cost})

        return batch_data_samples
