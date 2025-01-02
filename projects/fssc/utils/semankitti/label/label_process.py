import numpy as np
from tqdm import tqdm
import os
import glob
import io_data as SemanticKittiIO
import argparse
import pickle as pkl
from joblib import Parallel, delayed


def label_rectification(
    grid_ind, voxel_label, instance_label, dynamic_classes=[1, 4, 5, 6, 7, 8], voxel_shape=(256, 256, 32), ignore_class_label=255
):
    segmentation_label = voxel_label[grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]]

    for c in dynamic_classes:
        voxel_pos_class_c = voxel_label == c
        instance_label_class_c = instance_label[segmentation_label == c].squeeze(1)

        if len(instance_label_class_c) == 0:
            pos_to_remove = voxel_pos_class_c
        elif len(instance_label_class_c) > 0 and np.sum(voxel_pos_class_c) > 0:
            mask_class_c = np.zeros(voxel_shape, dtype=bool)
            point_pos_class_c = grid_ind[segmentation_label == c]
            uniq_instance_label_class_c = np.unique(instance_label_class_c)

            for i in uniq_instance_label_class_c:
                point_pos_instance_i = point_pos_class_c[instance_label_class_c == i]
                x_max, y_max, z_max = np.amax(point_pos_instance_i, axis=0)
                x_min, y_min, z_min = np.amin(point_pos_instance_i, axis=0)
                mask_class_c[x_min:x_max, y_min:y_max, z_min:z_max] = True

            pos_to_remove = voxel_pos_class_c & ~mask_class_c

        voxel_label[pos_to_remove] = ignore_class_label

    return voxel_label


def process_frame_for_generate_label(label_path, invalid_path, remap_lut, out_dir):
    frame_id, _ = os.path.splitext(os.path.basename(label_path))
    LABEL = SemanticKittiIO._read_label_SemKITTI(label_path)
    INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
    LABEL[np.isclose(INVALID, 1)] = 255
    LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)
    LABEL = LABEL.reshape([256, 256, 32])

    filename = frame_id + ".pkl"
    label_filename = os.path.join(out_dir, filename)
    with open(label_filename, "wb") as file:
        pkl.dump(LABEL, file)


def generate_label(config):
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    remap_lut = SemanticKittiIO._get_remap_lut(config.config_path)

    for sequence in sequences:
        sequence_path = os.path.join(config.kitti_root, "sequences", sequence)
        label_paths = sorted(glob.glob(os.path.join(sequence_path, "voxels", "*.label")))
        invalid_paths = sorted(glob.glob(os.path.join(sequence_path, "voxels", "*.invalid")))
        out_dir = os.path.join(config.kitti_preprocess_root, "ssc", sequence)
        os.makedirs(out_dir, exist_ok=True)

        Parallel(n_jobs=-1)(
            delayed(process_frame_for_generate_label)(label_paths[i], invalid_paths[i], remap_lut, out_dir) for i in tqdm(range(len(label_paths)))
        )


def process_frame_for_generate_rect_label(label_path, pc_path, pc_label, min_bound, max_bound, intervals, out_dir):
    frame_id, _ = os.path.splitext(os.path.basename(label_path))
    LABEL = np.load(label_path, allow_pickle=True)

    PC = SemanticKittiIO._read_pointcloud_SemKITTI(pc_path)[:, :3]
    PC_INSTANCE = np.fromfile(pc_label, dtype=np.uint32).reshape(-1, 1)

    box_filter = np.logical_and(
        np.logical_and(PC[:, 0] >= min_bound[0], PC[:, 0] < max_bound[0]),
        np.logical_and(PC[:, 1] >= min_bound[1], PC[:, 1] < max_bound[1]),
        np.logical_and(PC[:, 2] >= min_bound[2], PC[:, 2] < max_bound[2]),
    )
    PC = PC[box_filter]
    PC_INSTANCE = PC_INSTANCE[box_filter]
    grid_ind = (np.floor((np.clip(PC, min_bound, max_bound) - min_bound) / intervals)).astype(np.int32)

    LABEL_ds = label_rectification(grid_ind, LABEL.copy(), PC_INSTANCE)

    filename = frame_id + ".pkl"
    label_filename = os.path.join(out_dir, filename)
    with open(label_filename, "wb") as file:
        pkl.dump(LABEL_ds, file)


def generate_rect_label(config):
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    min_bound = np.array([0, -25.6, -2.0])
    max_bound = np.array([51.2, 25.6, 4.4])
    intervals = np.array([0.2, 0.2, 0.2])
    for sequence in sequences:
        assert os.path.exists((os.path.join(config.kitti_preprocess_root, "ssc", sequence, "*.pkl"))), "Please generate labels first"
        sequence_path = os.path.join(config.kitti_root, "dataset", "sequences", sequence)
        pc_paths = sorted(glob.glob(os.path.join(sequence_path, "velodyne", "*.bin")))
        pc_labels = sorted(glob.glob(os.path.join(sequence_path, "labels", "*.label")))
        label_paths = sorted(glob.glob(os.path.join(config.kitti_preprocess_root, "ssc", sequence, "*.pkl")))
        out_dir = os.path.join(config.kitti_preprocess_root, "ssc_rect", sequence)
        os.makedirs(out_dir, exist_ok=True)

        Parallel(n_jobs=-1)(
            delayed(process_frame_for_generate_rect_label)(label_paths[i], pc_paths[i], pc_labels[i], min_bound, max_bound, intervals, out_dir)
            for i in tqdm(range(len(label_paths)))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./label_preprocess.py")
    parser.add_argument("--kitti_root", "-r", type=str, required=True, help="kitti_root")
    parser.add_argument("--kitti_preprocess_root", "-p", type=str, required=True, help="kitti_preprocess_root")
    parser.add_argument("--config_path", "-c", type=str, required=True, help="kitti_root")
    parser.add_argument("--rect_label", action="store_true", help="Generate rectified labels")

    config, unparsed = parser.parse_known_args()
    generate_label(config) if not config.rect_label else generate_rect_label(config)
