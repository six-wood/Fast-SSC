import argparse
import numpy as np
import open3d as o3d
import os
import cv2


def unpack(compressed):
    """Given a bit-encoded voxel grid, make a normal voxel grid out of it."""
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1
    return uncompressed


def _read_SemKITTI(path, dtype, do_unpack=True):
    """Reads and optionally unpacks the binary data."""
    bin_data = np.fromfile(path, dtype=dtype)  # Flattened array
    if do_unpack:
        bin_data = unpack(bin_data)
    return bin_data


def _read_pred(path: str) -> np.array:
    label = _read_SemKITTI(path, dtype=np.uint16, do_unpack=False).astype(np.float32)
    return label


def _read_truth(path: str) -> np.array:
    label = np.load(path, allow_pickle=True)
    return label


validatation_set = "08"


def create_point_cloud(points, pc_rgb):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    pc.colors = o3d.utility.Vector3dVector(pc_rgb / 255)
    R = pc.get_rotation_matrix_from_xyz((-np.pi / 2.5, 0, np.pi / 3))  # np.pi/2 是 90 度的弧度值
    pc.rotate(R, center=(0, 0, 0))  # 应用旋转
    return pc


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the point cloud")
    parser.add_argument("--data-root", default="data/semantickitti", help="The root directory of the dataset")
    parser.add_argument("--pc-path", default=None, help="The path of the point cloud")
    args = parser.parse_args()
    return args


class SemanticKITTIProcessor:
    def __init__(self, data_root, validation_set):
        self.data_root = data_root
        self.validation_set = validation_set

        self.pc_dir = f"{data_root}/sequences/{validatation_set}/velodyne"
        self.pc_label_dir = f"{data_root}/sequences/{validatation_set}/labels"
        self.SSC_dir = f"{data_root}/ssc/{validatation_set}"
        self.SSC_rect_dir = f"{data_root}/ssc_rect/{validatation_set}"

        self.Fast_SSC_dir = f"{data_root}/ssc_cache/sequences/{validatation_set}/predictions"
        self.Fast_SC_dir = f"{data_root}/sc_cache/sequences/{validatation_set}/predictions"
        self.SSA_SC_dir = f"{data_root}/SSA_SC/sequences/{validatation_set}/predictions"
        self.SSC_RS_dir = f"{data_root}/SSC_RS/sequences/{validatation_set}/predictions"

        out_dir = f"{data_root}/show"
        self.pc_out_dir = f"{out_dir}/pc"
        self.ssc_out_dir = f"{out_dir}/ssc"
        self.bev_ssc_out_dir = f"{out_dir}/bev_ssc"
        self.fig1_out_dir = f"{out_dir}/fig1"
        self.fig2_out_dir = f"{out_dir}/fig2"
        self.fig3_out_dir = f"{out_dir}/fig3"

        self.fast_ssc_out_dir = f"{out_dir}/ssc_cache"
        self.fast_sc_out_dir = f"{out_dir}/sc_cache"
        self.ssa_sc_out_dir = f"{out_dir}/ssa_sc"
        self.ssc_rs_out_dir = f"{out_dir}/ssc_rs"
        self.ssc_rect_out_dir = f"{out_dir}/ssc_rect"

        os.makedirs(self.pc_out_dir, exist_ok=True)
        os.makedirs(self.ssc_out_dir, exist_ok=True)
        os.makedirs(self.bev_ssc_out_dir, exist_ok=True)
        os.makedirs(self.ssc_rect_out_dir, exist_ok=True)
        os.makedirs(self.fig1_out_dir, exist_ok=True)
        os.makedirs(self.fast_ssc_out_dir, exist_ok=True)
        os.makedirs(self.fast_sc_out_dir, exist_ok=True)
        os.makedirs(self.ssa_sc_out_dir, exist_ok=True)
        os.makedirs(self.ssc_rs_out_dir, exist_ok=True)
        os.makedirs(self.fig2_out_dir, exist_ok=True)
        os.makedirs(self.fig3_out_dir, exist_ok=True)

        self.labels_map = {
            0: 0,  # "unlabeled"
            1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 1,  # "car"
            11: 2,  # "bicycle"
            13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 3,  # "motorcycle"
            16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 4,  # "truck"
            20: 5,  # "other-vehicle"
            30: 6,  # "person"
            31: 7,  # "bicyclist"
            32: 8,  # "motorcyclist"
            40: 9,  # "road"
            44: 10,  # "parking"
            48: 11,  # "sidewalk"
            49: 12,  # "other-ground"
            50: 13,  # "building"
            51: 14,  # "fence"
            52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 9,  # "lane-marking" to "road" ---------------------------------mapped
            70: 15,  # "vegetation"
            71: 16,  # "trunk"
            72: 17,  # "terrain"
            80: 18,  # "pole"
            81: 19,  # "traffic-sign"
            99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 1,  # "moving-car" to "car" ------------------------------------mapped
            253: 7,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 6,  # "moving-person" to "person" ------------------------------mapped
            255: 8,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 5,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 4,  # "moving-truck" to "truck" --------------------------------mapped
            259: 5,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }

        self.palette = list(
            [
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
            ]
        )
        self.grid_size = [256, 256, 32]
        self.voxel_size = np.array([0.2, 0.2, 0.2])
        self.offset = np.array([0, -25.6, -2.0])
        self.pc_range = np.array([[0, 51.2], [-25.6, 25.6], [-2, 4.4]])

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1024, height=1024)

        render_option = self.vis.get_render_option()
        render_option.point_size = 1.0  # 设置点的大小，调整这个值来改变大小

        self.vis_voxel = o3d.visualization.Visualizer()
        self.vis_voxel.create_window(width=1024, height=1024)

        render_option = self.vis_voxel.get_render_option()
        render_option.point_size = 3.0  # 设置点的大小，调整这个值来改变大小

    def set_view(self, vis, front, lookat, up, zoom):
        ctr = vis.get_view_control()
        ctr.set_front(front=front)
        ctr.set_lookat(lookat=lookat)
        ctr.set_up(up=up)
        ctr.set_zoom(zoom)

    def render_and_save(self, pc, output_path):
        self.vis.clear_geometries()
        self.vis.add_geometry(pc)
        self.set_view(self.vis, [0, 0, 1], [0, 10, -20], [0, 1, 0], 0.5)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.capture_screen_image(output_path)

    def render_and_save_voxel(self, pc, output_path, voxel_size=3.0):
        render_option = self.vis_voxel.get_render_option()
        render_option.point_size = voxel_size  # 设置点的大小，调整这个值来改变大小
        self.vis_voxel.clear_geometries()
        self.vis_voxel.add_geometry(pc)
        self.set_view(self.vis_voxel, [0, 0, 1], [0, 10, -20], [0, 1, 0], 0.5)
        self.vis_voxel.poll_events()
        self.vis_voxel.update_renderer()
        self.vis_voxel.capture_screen_image(output_path)

    def label2rgb(self, pc, pc_label):
        pc_filter = np.logical_and.reduce(
            (
                pc[:, 0] > self.pc_range[0, 0],
                pc[:, 0] < self.pc_range[0, 1],
                pc[:, 1] > self.pc_range[1, 0],
                pc[:, 1] < self.pc_range[1, 1],
                pc[:, 2] > self.pc_range[2, 0],
                pc[:, 2] < self.pc_range[2, 1],
            )
        )
        pc = pc[pc_filter]
        pc_label = pc_label[pc_filter]
        pc_label = np.vectorize(self.labels_map.get)(pc_label)
        pc_rgb = np.zeros((pc_label.shape[0], 3), dtype=np.uint8)
        pc_rgb = np.array([self.palette[label] for label in pc_label])
        return pc, pc_rgb.astype(np.float32)

    def intensity2rgb(self, pc):
        """
        根据反射强度生成点云的颜色。

        参数:
        - pc: np.ndarray, 点云坐标 (N, 3)。
        - intensity: np.ndarray, 点云反射强度 (N,)。

        返回:
        - pc_filtered: np.ndarray, 过滤后的点云坐标。
        - pc_rgb: np.ndarray, 每个点的 RGB 颜色值 (N, 3)。
        """
        # 定义点云范围 (可根据实际情况修改)
        pc_filter = np.logical_and.reduce(
            (
                pc[:, 0] > self.pc_range[0, 0],
                pc[:, 0] < self.pc_range[0, 1],
                pc[:, 1] > self.pc_range[1, 0],
                pc[:, 1] < self.pc_range[1, 1],
                pc[:, 2] > self.pc_range[2, 0],
                pc[:, 2] < self.pc_range[2, 1],
            )
        )
        pc = pc[pc_filter]
        intensity = pc[:, 3]
        intensity = (intensity * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(intensity, cv2.COLORMAP_JET)

        return pc, colormap.reshape(colormap.shape[0], -1).astype(np.float32)[:, ::-1]

    def voxel2pc(self, voxel: np.array, map: bool = False) -> np.array:
        """
        Args:
            voxel: np.array, shape=(N, N, N, C)
            grid_size: float, the size of the voxel grid
            voxel_size: float, the size of the voxel
        Returns:
            pc: np.array, shape=(N, N, N, C)
        """
        # Reshape the grid
        voxel = voxel.reshape(self.grid_size)

        # Generate grid coordinates
        x, y, z = np.meshgrid(
            np.arange(self.grid_size[0]),
            np.arange(self.grid_size[1]),
            np.arange(self.grid_size[2]),
            indexing="ij",
        )

        # Flatten and filter out unoccupied voxels
        pc = np.vstack((x.ravel(), y.ravel(), z.ravel(), voxel.ravel())).T
        pc_valid = pc[(pc[:, 3] > 0) & (pc[:, 3] < 255)].astype(np.float32)

        # Convert to point cloud
        pc_valid[:, :3] = pc_valid[:, :3] * self.voxel_size + self.offset
        pc_valid_label = pc_valid[:, 3].astype(np.int32)
        # if map:
        pc_valid_label = np.vectorize(self.labels_map.get)(pc_valid_label) if map else pc_valid_label
        pc_rgb = np.zeros((pc_valid.shape[0], 3), dtype=np.uint8)
        for label_index in range(20):
            pc_rgb[pc_valid_label == label_index] = self.palette[label_index]
        pc_valid = np.concatenate([pc_valid[:, :3], pc_rgb], axis=1)

        return pc_valid

    def process(self, pc_path=None):
        paths = sorted(os.listdir(self.pc_dir)) if pc_path is None else [pc_path]
        for pc_path in paths:
            pc_path = os.path.join(self.pc_dir, pc_path)
            pc_label_path = os.path.join(self.pc_label_dir, pc_path.split("/")[-1].replace(".bin", ".label"))

            SSC_path = os.path.join(self.SSC_dir, pc_path.split("/")[-1].replace(".bin", ".pkl"))
            SSC_rect_path = os.path.join(self.SSC_rect_dir, pc_path.split("/")[-1].replace(".bin", ".pkl"))

            Fast_SSC_path = os.path.join(self.Fast_SSC_dir, pc_path.split("/")[-1].replace(".bin", ".label"))
            Fast_SC_path = os.path.join(self.Fast_SC_dir, pc_path.split("/")[-1].replace(".bin", ".label"))
            SSA_SC_path = os.path.join(self.SSA_SC_dir, pc_path.split("/")[-1].replace(".bin", ".label"))
            SSC_RS_path = os.path.join(self.SSC_RS_dir, pc_path.split("/")[-1].replace(".bin", ".label"))

            pc_out_path = os.path.join(self.pc_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))
            ssc_out_path = os.path.join(self.ssc_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))
            bev_ssc_out_path = os.path.join(self.bev_ssc_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))
            ssc_rect_out_path = os.path.join(self.ssc_rect_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))
            fig1 = os.path.join(self.fig1_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))

            fast_ssc_out_path = os.path.join(self.fast_ssc_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))
            fast_sc_out_path = os.path.join(self.fast_sc_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))
            ssa_sc_out_path = os.path.join(self.ssa_sc_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))
            ssc_rs_out_path = os.path.join(self.ssc_rs_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))
            fig2 = os.path.join(self.fig2_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))
            fig3 = os.path.join(self.fig3_out_dir, pc_path.split("/")[-1].replace(".bin", ".png"))

            pc = _read_SemKITTI(pc_path, dtype=np.float32, do_unpack=False).reshape(-1, 4)
            # pc_label = _read_SemKITTI(pc_label_path, dtype=np.uint32, do_unpack=False) & 0xFFFF
            pc, pc_rgb = self.intensity2rgb(pc)
            pc_o3d = create_point_cloud(pc, pc_rgb)
            self.render_and_save(pc_o3d, pc_out_path)
            pc_img = cv2.imread(pc_out_path)
            pc_img = pc_img[256 + 128 : 1024 - 128, 128:1024]
            cv2.imwrite(pc_out_path, pc_img)

            SSC = _read_truth(SSC_path)
            SSC = self.voxel2pc(SSC)
            SSC_o3d = create_point_cloud(SSC[:, :3], SSC[:, 3:])
            self.render_and_save_voxel(SSC_o3d, ssc_out_path)
            # self.render_and_save_voxel(SC_o3d_, ssc_out_path)
            SSC_img = cv2.imread(ssc_out_path)
            SSC_img = SSC_img[256 + 128 : 1024 - 128, 128:1024]
            cv2.imwrite(ssc_out_path, SSC_img)

            SSC = _read_truth(SSC_path)
            SSC_shape = SSC.shape
            SSC_bev_img = np.zeros((SSC_shape[0], SSC_shape[1]), dtype=np.uint8)
            SSC_bev_img_rgb = np.zeros((SSC_shape[0], SSC_shape[1], 3), dtype=np.uint8)
            SSC_voxel = np.where((SSC > 0) & (SSC < 255), SSC, 0)
            SSC_bev_img = self.generate_bev_image(SSC_shape, SSC_voxel)
            SSC_bev_img_rgb = np.array([self.palette[label] for label in SSC_bev_img.ravel()]).reshape(SSC_shape[0], SSC_shape[1], 3).astype(np.uint8)
            SSC_bev_img_rgb = cv2.cvtColor(SSC_bev_img_rgb, cv2.COLOR_BGR2RGB)
            SSC_bev_img_rgb = cv2.flip(cv2.flip(SSC_bev_img_rgb, 0), 1)
            cv2.imwrite(bev_ssc_out_path, SSC_bev_img_rgb)

            SSC_rect = _read_truth(SSC_rect_path)
            SSC_rect = self.voxel2pc(SSC_rect)
            SSC_rect_o3d = create_point_cloud(SSC_rect[:, :3], SSC_rect[:, 3:])
            self.render_and_save_voxel(SSC_rect_o3d, ssc_rect_out_path)
            SSC_rect_img = cv2.imread(ssc_rect_out_path)
            SSC_rect_img = SSC_rect_img[256 + 128 : 1024 - 128, 128:1024]
            cv2.imwrite(ssc_rect_out_path, SSC_rect_img)

            # cat
            cat_img = np.concatenate((pc_img, SSC_img), axis=0)
            cv2.imshow("PC+SSC", cat_img)
            cv2.imwrite(fig1, cat_img)

            # cat
            cat_img = np.concatenate((SSC_img, SSC_rect_img), axis=0)
            cv2.imshow("SSC+SSC_rect", cat_img)
            cv2.imwrite(fig3, cat_img)

            Fast_SSC = _read_pred(Fast_SSC_path).reshape(-1, 1)
            Fast_SSC = self.voxel2pc(Fast_SSC, True)
            Fast_SSC_o3d = create_point_cloud(Fast_SSC[:, :3], Fast_SSC[:, 3:])
            self.render_and_save_voxel(Fast_SSC_o3d, fast_ssc_out_path)
            Fast_SSC_img = cv2.imread(fast_ssc_out_path)
            Fast_SSC_img = Fast_SSC_img[256 + 128 : 1024 - 128, 128:1024]
            cv2.imwrite(fast_ssc_out_path, Fast_SSC_img)

            Fast_SC = _read_pred(Fast_SC_path).reshape(-1, 1)
            Fast_SC = self.voxel2pc(Fast_SC, True)
            Fast_SC_o3d = create_point_cloud(Fast_SC[:, :3], 100 * np.ones_like(Fast_SC[:, 3:]))
            self.render_and_save_voxel(Fast_SC_o3d, fast_sc_out_path, 1.0)
            Fast_SC_img = cv2.imread(fast_sc_out_path)
            Fast_SC_img = Fast_SC_img[256 + 128 : 1024 - 128, 128:1024]
            cv2.imwrite(fast_sc_out_path, Fast_SC_img)
            cat_img = np.concatenate((Fast_SSC_img, Fast_SC_img), axis=1)
            cv2.imshow("Fast_SSC+Fast_SC", cat_img)

            SSA_SC = _read_pred(SSA_SC_path).reshape(-1, 1)
            SSA_SC = self.voxel2pc(SSA_SC, True)
            SSA_SC_o3d = create_point_cloud(SSA_SC[:, :3], SSA_SC[:, 3:])
            self.render_and_save_voxel(SSA_SC_o3d, ssa_sc_out_path)
            SSA_SC_img = cv2.imread(ssa_sc_out_path)
            SSA_SC_img = SSA_SC_img[256 + 128 : 1024 - 128, 128:1024]
            cv2.imwrite(ssa_sc_out_path, SSA_SC_img)

            SSC_RS = _read_pred(SSC_RS_path).reshape(-1, 1)
            SSC_RS = self.voxel2pc(SSC_RS, True)
            SSC_RS_o3d = create_point_cloud(SSC_RS[:, :3], SSC_RS[:, 3:])
            self.render_and_save_voxel(SSC_RS_o3d, ssc_rs_out_path)
            SSC_RS_img = cv2.imread(ssc_rs_out_path)
            SSC_RS_img = SSC_RS_img[256 + 128 : 1024 - 128, 128:1024]
            cv2.imwrite(ssc_rs_out_path, SSC_RS_img)

            # cat
            cat_img = np.concatenate((SSC_img, Fast_SSC_img, SSA_SC_img, SSC_RS_img), axis=1)
            cv2.imshow("SSC+Fast_SSC+SSA_SC+SSC_RS", cat_img)
            cv2.imwrite(fig2, cat_img)

            cv2.waitKey(1)
            print("data processed: ", pc_path.split("/")[-1])
        self.vis.destroy_window()

    def generate_bev_image(self, SSC_shape, SSC_voxel):
        voxel_flip = np.flip(SSC_voxel, axis=2)  # 沿第三个维度翻转数组，通常是为了在俯视（BEV）视图中获得符合需求的深度顺序。
        mask = voxel_flip > 0  # 创建一个布尔掩码，标记哪些位置的体素值大于零，表示该处存在有效信息。
        idx = np.argmax(mask, axis=2)  # 沿第三个维度查找布尔值第一次出现 True 的位置，得到在顶视角最外层（翻转后靠近观测者）的体素索引。
        exists = np.any(mask, axis=2)  # 判断在每个 (x, y) 坐标上是否存在任意大于零的体素。
        SSC_bev_img = np.where(
            exists, voxel_flip[np.arange(SSC_shape[0])[:, None], np.arange(SSC_shape[1])[None, :], idx], 0
        )  # 若 (x, y) 坐标存在有效体素，则根据前面求出的索引从翻转后的体素中取该体素值，否则填 0，最终得到俯视图图像。
        return SSC_bev_img


if __name__ == "__main__":
    args = parse_args()
    pc_path = args.pc_path
    processor = SemanticKITTIProcessor(data_root=args.data_root, validation_set=validatation_set)
    processor.process(pc_path)
