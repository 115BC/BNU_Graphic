# config.py
import torch
import math

class Config:
    # ---------- 路径 ----------
    target_mesh_path = "data/cow.obj"
    output_dir = "results"

    # ---------- 优化参数 ----------
    num_iterations = 6000               # 增加迭代次数
    lr = 1e-3                           # 学习率
    lr_scheduler = "cosine"             # 余弦退火
    save_interval = 200

    # ---------- 渲染参数 ----------
    image_size = 256
    sigma = 1e-6                        # 更小的值使边缘更锐利，但保持梯度
    gamma = 1e-6

    # ---------- 摄像机配置 ----------
    num_views = 32                      # 32个视角
    distance = 2.0

    # ---------- 源网格 ----------
    source_mesh_level = 5               # 高分辨率球体

    # ---------- 正则化权重 (动态调节，起始值) ----------
    w_laplacian_start = 0.3
    w_laplacian_end = 0.05
    w_edge_start = 0.2
    w_edge_end = 0.02
    w_normal_start = 0.05
    w_normal_end = 0.01

    # ---------- 损失类型 ----------
    silhouette_loss_type = "dice_l1"    # 混合损失

    # ---------- 设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_camera_poses(cls):
        """生成多样化的摄像机视角：水平环 + 上下倾斜 + 顶部/底部"""
        poses = []
        # 水平环 (俯仰0°) 每30度一个，共12个
        for az in range(0, 360, 30):
            poses.append((az, 0, cls.distance))
        # 上倾斜 20° 和 40°，每60度一个 (避免过多)
        for el in [20, 40]:
            for az in range(0, 360, 60):
                poses.append((az, el, cls.distance))
        # 下倾斜 -20°，每60度一个
        for az in range(0, 360, 60):
            poses.append((az, -20, cls.distance))
        # 顶部视角 75°，每120度一个
        for az in range(0, 360, 120):
            poses.append((az, 75, cls.distance))
        # 截取前 cls.num_views 个
        return poses[:cls.num_views]

    @classmethod
    def get_regularization_weights(cls, iteration):
        """根据当前迭代次数，线性插值正则化权重"""
        t = iteration / cls.num_iterations
        # 从起始值线性衰减到结束值
        w_lap = cls.w_laplacian_start * (1 - t) + cls.w_laplacian_end * t
        w_edge = cls.w_edge_start * (1 - t) + cls.w_edge_end * t
        w_norm = cls.w_normal_start * (1 - t) + cls.w_normal_end * t
        return w_lap, w_edge, w_norm