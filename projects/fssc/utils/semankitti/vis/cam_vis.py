
"""Use the pytorch-grad-cam tool to visualize Class Activation Maps (CAM).

requirement: pip install grad-cam
"""

import numpy as np
import pickle
import torch
import tqdm
import cv2
import os
from argparse import ArgumentParser
from typing import List
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from mmdet3d.apis import init_model
from mmdet3d.datasets import LoadPointsFromFile
import types

class_names = [
    "free",  # 0
    "car",  # 1
    "bicycle",  # 2
    "motorcycle",  # 3
    "truck",  # 4
    "other-vehicle",  # 5
    "person",  # 6
    "bicyclist",  # 7
    "motorcyclist",  # 8
    "road",  # 9
    "parking",  # 10
    "sidewalk",  # 11
    "other-ground",  # 12
    "building",  # 13
    "fence",  # 14
    "vegetation",  # 15
    "trunk",  # 16
    "terrian",  # 17
    "pole",  # 18
    "traffic-sign",  # 19
]


class CustomCAM(GradCAM):
    """Custom GradCAM for computing Class Activation Maps (CAM)."""

    def compute_cam_per_layer(self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool) -> np.ndarray:
        """Compute the CAM for each layer."""
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = activations_list[i] if i < len(activations_list) else None
            layer_grads = grads_list[i] if i < len(grads_list) else None

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size[-2:])  # BEV YX
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer


class SemanticSegmentationTarget:
    """Wraps the model to compute class activation map based on category and mask."""

    def __init__(self, category, mask):
        self.category = category
        self.mask = mask

    def __call__(self, model_output):
        """Compute the sum of masked output."""
        return (model_output[self.category, :, :] * self.mask).sum()


class PointCloudProcessor:
    """Processes point cloud data and generates CAM for multiple categories."""

    def __init__(self, model, target_layers: List[str], category_list: List[int], out_dir: str):
        self.model = model
        self.target_layers = target_layers
        self.category_list = category_list
        self.load_pts = LoadPointsFromFile(coord_type="LIDAR", load_dim=4, use_dim=4)
        self.cam = CustomCAM(model=model, target_layers=target_layers)
        self.out_dir = out_dir  # Path to save the CAM images

    def process_single_point_cloud(self, data, idx: int) -> None:
        """Process a single point cloud and save CAMs for each category immediately."""
        data["lidar_points"]["lidar_path"] = os.path.join(self.model.cfg["val_split"]["data_root"], data["lidar_points"]["lidar_path"])
        pts = self.load_pts(data)["points"].tensor.cuda()
        voxel_dict = self.model.voxelize([pts])
        geo_fea = self.model.pts_voxel_encoder(voxel_dict["voxels"], voxel_dict["coors"]).permute(0, 1, 4, 3, 2)

        with torch.no_grad():
            ssc = self.model(geo_fea).argmax(dim=1)

        for category_index in self.category_list:
            cam_image = self.generate_cam_for_category(category_index, ssc, geo_fea, voxel_dict)
            self.save_cam_image(cam_image, category_index, idx)

    def generate_cam_for_category(self, category_index: int, ssc: torch.Tensor, geo_fea: torch.Tensor, voxel_dict: dict) -> np.ndarray:
        """Generate CAM for a specific category."""
        mask_float = (ssc == category_index).float()
        input_tensor = geo_fea.clone()

        bev_image = torch.zeros(self.model.voxel_layer.grid_shape[0], self.model.voxel_layer.grid_shape[1], device=input_tensor.device, dtype=torch.float32)
        bev_image[voxel_dict["coors"][:, 2], voxel_dict["coors"][:, 1]] = voxel_dict["voxels"][:, 3]
        bev_image = (bev_image.cpu().numpy() * 255).astype(np.uint8)
        bev_mask = bev_image != 0
        bev_image = (cv2.applyColorMap(bev_image, cv2.COLORMAP_JET) * bev_mask[:, :, None]).astype(np.float32) / 255.0

        # Generate Grad CAM
        targets = [SemanticSegmentationTarget(category_index, mask_float)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(bev_image, grayscale_cam, image_weight=0.65)

        return cv2.transpose(cv2.flip(cam_image, 1))

    def save_cam_image(self, cam_image: np.ndarray, category_index: int, idx: int) -> None:
        """Save the CAM image for a given category."""
        # Ensure the directory exists
        category_dir = os.path.join(self.out_dir, class_names[category_index])

        # Save the image
        cam_filename = os.path.join(category_dir, str(idx).zfill(6) + ".jpg")
        cv2.imwrite(cam_filename, cam_image)


class ModelHandler:
    """Handles the initialization, forward pass, and point cloud processing."""

    def __init__(self, config: str, checkpoint: str, device: str, out_dir: str):
        self.model = init_model(config, checkpoint, device=device)
        self.model.forward = types.MethodType(forward, self.model)
        self.out_dir = out_dir  # Path to save the CAM images

    def load_data_list(self, ann_file: str) -> List[dict]:
        """Load the data list from annotation file."""
        ann_info = pickle.load(open(ann_file, "rb"))
        return ann_info["data_list"]

    def process_point_clouds(self, data_list: List[dict], category_list: List[int], target_layers: List[str]) -> None:
        """Process a list of point clouds and save CAMs for each category."""
        processor = PointCloudProcessor(self.model, target_layers, category_list, self.out_dir)

        for idx, data in tqdm.tqdm(enumerate(data_list), total=len(data_list)):
            processor.process_single_point_cloud(data, idx)


def forward(self, geo_fea: torch.Tensor) -> dict:
    """Forward function for point cloud data."""
    geo_fea = geo_fea.permute(0, 1, 4, 3, 2)  # BC xyz->BC zyx
    geo_fea = self.sparse_backbone(geo_fea)
    bev_fea = self.extract_bev_feat(geo_fea)
    bev_fea = self.neck(bev_fea)

    ssc = self.ssc_head(bev_fea) if self.ssc_head is not None else None
    return ssc


def main():
    parser = ArgumentParser()
    parser.add_argument("pts", help="Point cloud file(s), either a single file or a directory of files")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--out-dir", default="cam.jpg", help="Path to output cam directory")
    parser.add_argument("--target-layers", default="backbone.layer4[2]", help="Target layers to visualize CAM")
    parser.add_argument("--category", nargs="+", type=int, default=[7], help="List of categories to visualize CAM")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    args = parser.parse_args()

    # Initialize the model handler
    model_handler = ModelHandler(config=args.config, checkpoint=args.checkpoint, device=args.device, out_dir=args.out_dir)

    # Load point clouds
    ann_file = os.path.join(model_handler.model.cfg["val_split"]["data_root"], model_handler.model.cfg["val_split"]["ann_file"])
    data_list = model_handler.load_data_list(ann_file)
    # Save or display the CAM results
    os.makedirs(args.out_dir, exist_ok=True)
    [os.makedirs(os.path.join(args.out_dir, class_names[class_index]), exist_ok=True) for class_index in args.category]
    # Process point clouds to generate CAM for multiple categories
    target_layers = [eval(f"model_handler.model.{args.target_layers}")]
    with torch.cuda.amp.autocast():
        model_handler.process_point_clouds(data_list, args.category, target_layers)


if __name__ == "__main__":
    main()
