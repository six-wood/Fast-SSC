import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet.datasets.transforms import RandomFlip


@TRANSFORMS.register_module()
class LoadSscLabelFromFile(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: dict) -> dict:
        results["voxel_label"] = np.load(results["voxel_label_path"], allow_pickle=True)
        return results


@TRANSFORMS.register_module()
class RandomFlipOcc(RandomFlip):
    """Flip the points & voxel_label.

    Required Keys:

    - points (np.float32)
    - voxel_label (np.float32)


    Args:

        ratio (float): The flipping probability
            in horizontal direction. Defaults to 0.0.

    """

    def __init__(self, ratio: float = 0.0, **kwargs) -> None:
        super(RandomFlipOcc, self).__init__(prob=ratio, direction="horizontal", **kwargs)
        self.ratio = ratio

        if ratio is not None:
            assert isinstance(ratio, (int, float)) and 0 <= ratio <= 1

    def random_flip_data_3d(self, input_dict: dict, direction: str = "horizontal") -> None:
        """Flip 3D data randomly.

        `random_flip_data_3d` should take these situations into consideration:

        - 1. LIDAR-based 3d detection
        - 2. LIDAR-based 3d segmentation
        - 3. vision-only detection
        - 4. multi-modality 3d detection.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Defaults to 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
            updated in the result dict.
        """
        assert direction in ["horizontal"]

        input_dict["points"].flip(direction)
        input_dict["voxel_label"] = np.flip(input_dict["voxel_label"], axis=1)

    def transform(self, input_dict: dict) -> dict:
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
            'pcd_horizontal_flip' keys are added
            into result dict.
        """

        if "pcd_horizontal_flip" not in input_dict:
            flip_horizontal = True if np.random.rand() < self.ratio else False
            input_dict["pcd_horizontal_flip"] = flip_horizontal

        if "transformation_3d_flow" not in input_dict:
            input_dict["transformation_3d_flow"] = []

        if input_dict["pcd_horizontal_flip"]:
            self.random_flip_data_3d(input_dict, "horizontal")
            input_dict["transformation_3d_flow"].extend(["HF"])

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f" flip_ratio_bev_horizontal={self.ratio})"
        return repr_str
