from numpy.linalg import inv
from os import path as osp
from pathlib import Path
import numpy as np
import mmengine
import os

fold_split = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "val": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
}
split_list = ["train", "valid", "test"]


def read_calib(calib_path):
    """
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            if line == "\n":
                break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    # 3x4 projection matrix for left camera
    calib_out["P2"] = calib_all["P2"].reshape(3, 4)
    calib_out["Tr"] = np.identity(4)  # 4x4 matrix
    calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
    return calib_out


def parse_poses(filename, calibration):
    """read poses file with per-scan poses from given filename

    Returns
    -------
    list
        list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


pose_dict = dict()


def get_semantickitti_info(root_path, split):
    """Create info file in the form of
    data_infos={
        'metainfo': {'DATASET': 'SemanticKITTI'},
        'data_list': {
            00000: {
                'lidar_points':{
                    'lidat_path':'sequences/00/velodyne/000000.bin'
                },
                'pts_semantic_mask_path':
                    'sequences/000/labels/000000.labbel',
                'sample_id': '00'
            },
            ...
        }
    }
    """

    data_infos = dict()
    data_infos["metainfo"] = dict(DATASET="SemanticKITTI")
    data_list = []
    for folder in fold_split[split]:

        files = (
            os.listdir(osp.join(root_path, "sequences", folder, "voxels"))
            if split == "test"
            else os.listdir(osp.join(root_path, "ssc", folder))
        )
        for file in files:
            file_name = file.split("/")[-1].split(".")[0]
            info_dict = dict()
            info_dict["lidar_points"] = dict(
                lidar_path=osp.join(
                    "sequences", folder, "velodyne", file_name + ".bin"
                ),
                voxel_size=0.2,
                num_pts_feats=4,
            )
            info_dict["voxel_label_path"] = (
                osp.join(
                    "ssc",
                    folder,
                    file_name + ".pkl",
                )
                if split != "test"
                else None
            )
            info_dict["pts_semantic_mask_path"] = (
                osp.join(
                    "sequences",
                    folder,
                    "labels",
                    file_name + ".label",
                )
                if split != "test"
                else None
            )
            info_dict["sample_id"] = folder + file_name
            data_list.append(info_dict)

    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_semantickitti_info_file(root_path, pkl_prefix, save_path, sweep=10):
    """Create info file of SemanticKITTI dataset.

    Directly generate info file without raw data.

    Args:
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
    """
    print("Generate info.")

    for split in ["train", "val", "test"]:
        for i_folder in fold_split[split]:
            pose_path = osp.join(
                root_path,
                "sequences",
                i_folder,
                "poses.txt",
            )
            calib = read_calib(
                osp.join(
                    root_path,
                    "sequences",
                    i_folder,
                    "calib.txt",
                )
            )
            pose_dict[str(i_folder)] = parse_poses(pose_path, calib)

    save_path = Path(save_path)

    semantickitti_infos_train = get_semantickitti_info(
        root_path=root_path, split="train"
    )
    filename = save_path / f"{pkl_prefix}_infos_train.pkl"
    print(f"SemanticKITTI info train file is saved to {filename}")
    mmengine.dump(semantickitti_infos_train, filename)
    semantickitti_infos_val = get_semantickitti_info(root_path=root_path, split="val")
    filename = save_path / f"{pkl_prefix}_infos_val.pkl"
    print(f"SemanticKITTI info val file is saved to {filename}")
    mmengine.dump(semantickitti_infos_val, filename)
    semantickitti_infos_test = get_semantickitti_info(root_path=root_path, split="test")
    filename = save_path / f"{pkl_prefix}_infos_test.pkl"
    print(f"SemanticKITTI info test file is saved to {filename}")
    mmengine.dump(semantickitti_infos_test, filename)
