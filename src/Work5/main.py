import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
)
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

# ---------- 配置参数 ----------
NUM_ITERATIONS = 2000
LEARNING_RATE = 1.0                # SGD 初始学习率
MOMENTUM = 0.9                     # SGD 动量
SAVE_INTERVAL = 20

IMAGE_SIZE = 256
SIGMA = 1e-4                       # 软光栅化边缘模糊的 sigma 参数
GAMMA = 1e-4                       # 软阴影背景参数 (未直接使用，但保留)
FACES_PER_PIXEL = 50               # 官方推荐值

NUM_VIEWS = 36                     # 总视角数
NUM_VIEWS_PER_ITER = 2             # 随机采样视角数
DISTANCE = 2.7

SOURCE_MESH_LEVEL = 4              # 初始球体细分等级

W_EDGE = 1.0                       # 边长惩罚权重
W_LAPLACIAN = 1.0                  # 拉普拉斯平滑权重
W_NORMAL = 0.01                    # 法线一致性权重

OUTPUT_DIR = "results"
TARGET_MESH_PATH = "data/cow.obj"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- 1. 加载并规范化目标网格 ----------
def load_and_normalize_mesh(path):
    verts, faces, _ = load_obj(path)
    verts = verts.to(device)
    faces = faces.verts_idx.to(device)

    # 中心对齐
    center = verts.mean(dim=0)
    verts = verts - center

    # 缩放到最大坐标绝对值为 1
    max_coord = verts.abs().max()
    verts = verts / max_coord

    mesh = Meshes(verts=[verts], faces=[faces])
    return mesh

target_mesh = load_and_normalize_mesh(TARGET_MESH_PATH)
print(f"Target mesh: vertices={target_mesh.verts_packed().shape[0]}, faces={target_mesh.faces_packed().shape[0]}")

# ---------- 2. 初始化相机 ----------
def generate_cameras(num_views, distance):
    cameras = []
    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views
        R = euler_angles_to_matrix(torch.tensor([[0.0, azimuth, 0.0]]).to(device), "XYZ")
        T = torch.tensor([[0.0, 0.0, distance]]).to(device)
        cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60.0, znear=0.1, zfar=10.0)
        cameras.append(cam)
    return cameras

cameras = generate_cameras(NUM_VIEWS, DISTANCE)
print(f"Generated {len(cameras)} camera views.")

# ---------- 3. 渲染器设置 ----------
# 动态计算 blur_radius (官方公式)
alpha = 1e-4
blur_radius = np.log(1.0 / alpha - 1.0) * SIGMA

raster_settings = RasterizationSettings(
    image_size=IMAGE_SIZE,
    blur_radius=blur_radius,
    faces_per_pixel=FACES_PER_PIXEL,
)
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras[0], raster_settings=raster_settings),
    shader=SoftSilhouetteShader(),
)

def render_silhouettes(mesh, cameras, renderer):
    silhouettes = []
    for cam in cameras:
        fragments = renderer.rasterizer(mesh, cameras=cam)
        sil = renderer.shader(fragments, mesh, cameras=cam)[..., 3]
        silhouettes.append(sil.squeeze(0))
    return torch.stack(silhouettes, dim=0)

print("Generating target silhouettes...")
target_silhouettes = render_silhouettes(target_mesh, cameras, silhouette_renderer)
print(f"Target silhouettes shape: {target_silhouettes.shape}")

# ---------- 4. 初始化源网格 (球体) ----------
source_mesh = ico_sphere(SOURCE_MESH_LEVEL, device=device)
verts = source_mesh.verts_packed()
scale = verts.abs().max()
verts = verts / scale
source_mesh = Meshes(verts=[verts], faces=[source_mesh.faces_packed()])
print(f"Source mesh: vertices={source_mesh.verts_packed().shape[0]}, faces={source_mesh.faces_packed().shape[0]}")

# 可优化顶点
deform_verts = source_mesh.verts_packed().clone().detach().requires_grad_(True)
faces = source_mesh.faces_packed()
optimizer = torch.optim.SGD([deform_verts], lr=LEARNING_RATE, momentum=MOMENTUM)

# ---------- 辅助函数: 保存网格和对比图 ----------
def save_progress(iteration, current_mesh, current_silhouettes):
    # 保存网格
    save_path = os.path.join(OUTPUT_DIR, f"mesh_iter_{iteration:04d}.obj")
    save_obj(save_path, verts=current_mesh.verts_packed(), faces=current_mesh.faces_packed())

    # 保存剪影对比图
    nview = min(4, current_silhouettes.shape[0])
    fig, axes = plt.subplots(2, nview, figsize=(nview * 2, 4))
    for i in range(nview):
        # 修复：添加 .detach() 避免梯度问题
        axes[0, i].imshow(target_silhouettes[i].cpu().numpy(), cmap="gray")
        axes[0, i].set_title(f"Target {i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(current_silhouettes[i].detach().cpu().numpy(), cmap="gray")
        axes[1, i].set_title(f"Current {i}")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"silhouette_iter_{iteration:04d}.png"))
    plt.close()

def compute_iou(pred, target, thresh=0.5):
    pred_bin = (pred > thresh).float()
    target_bin = (target > thresh).float()
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# ---------- 5. 训练循环 ----------
print("Starting optimization...")
pbar = tqdm(range(1, NUM_ITERATIONS + 1))
for iteration in pbar:
    # 随机采样视角
    selected_indices = random.sample(range(NUM_VIEWS), NUM_VIEWS_PER_ITER)
    selected_cameras = [cameras[i] for i in selected_indices]
    targets_selected = target_silhouettes[selected_indices]
    current_mesh = Meshes(verts=[deform_verts], faces=[faces])
    current_silhouettes = render_silhouettes(current_mesh, selected_cameras, silhouette_renderer)

    # 计算损失
    loss_silhouette = torch.nn.functional.mse_loss(current_silhouettes, targets_selected)
    loss_edge = mesh_edge_loss(current_mesh)
    loss_laplacian = mesh_laplacian_smoothing(current_mesh, method="uniform")
    loss_normal = mesh_normal_consistency(current_mesh)
    loss_reg = W_EDGE * loss_edge + W_LAPLACIAN * loss_laplacian + W_NORMAL * loss_normal
    total_loss = loss_silhouette + loss_reg

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 监控与日志
    if iteration % 100 == 0:
        # 每100步计算一次所有视角的 IOU
        all_sils = render_silhouettes(current_mesh, cameras, silhouette_renderer)
        iou = compute_iou(all_sils, target_silhouettes)
        print(f"Iter {iteration}: Silhouette Loss={loss_silhouette.item():.4f}, "
              f"Edge Loss={loss_edge.item():.4f}, Laplacian Loss={loss_laplacian.item():.4f}, "
              f"Normal Loss={loss_normal.item():.4f}, IOU={iou:.4f}")
        pbar.set_postfix({"Sil": f"{loss_silhouette.item():.4f}", "IOU": f"{iou:.4f}"})
    else:
        pbar.set_postfix({"Sil": f"{loss_silhouette.item():.4f}"})

    if iteration % SAVE_INTERVAL == 0 or iteration == NUM_ITERATIONS:
        # 保存完整状态
        final_mesh = Meshes(verts=[deform_verts], faces=[faces])
        full_sils = render_silhouettes(final_mesh, cameras, silhouette_renderer)
        save_progress(iteration, final_mesh, full_sils)

print("Optimization finished!")