
import argparse
from semantickitti import semantickitti_converter


def semantickitti_data_prep(root_path, info_prefix, out_dir, sweep=10):
    """Prepare the info file for SemanticKITTI dataset.

    Args:
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
    """
    semantickitti_converter.create_semantickitti_info_file(root_path, info_prefix, out_dir, sweep=sweep)


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="./data/kitti",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)
parser.add_argument(
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument(
    "--with-plane",
    action="store_true",
    help="Whether to use plane information for kitti.",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./data/kitti",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--workers", type=int, default=4, help="number of threads to be used")
parser.add_argument(
    "--only-gt-database",
    action="store_true",
    help="""Whether to only generate ground truth database.
        Only used when dataset is NuScenes or Waymo!""",
)
parser.add_argument(
    "--skip-cam_instances-infos",
    action="store_true",
    help="""Whether to skip gathering cam_instances infos.
        Only used when dataset is Waymo!""",
)
parser.add_argument(
    "--skip-saving-sensor-data",
    action="store_true",
    help="""Whether to skip saving image and lidar.
        Only used when dataset is Waymo!""",
)
args = parser.parse_args()

if __name__ == "__main__":
    from mmengine.registry import init_default_scope

    init_default_scope("mmdet3d")
    if args.dataset == "semantickitti":
        semantickitti_data_prep(root_path=args.root_path, info_prefix=args.extra_tag, out_dir=args.out_dir, sweep=args.max_sweeps)
    else:
        raise NotImplementedError(f"Don't support {args.dataset} dataset.")
