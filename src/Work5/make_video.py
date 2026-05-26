# make_video.py
import os
import torch
import imageio
import numpy as np
from tqdm import tqdm
import glob
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
)

# ---------- 配置参数 ----------
OUTPUT_DIR = "results"
VIDEO_NAME = "deformation.mp4"
FPS = 30               # 帧率
RENDER_SIZE = 512      # 渲染分辨率
CAM_DISTANCE = 2.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定的相机视角
cam = FoVPerspectiveCameras(device=device, fov=60.0, znear=0.1, zfar=10.0)
cam.R = torch.eye(3).unsqueeze(0).to(device)
cam.T = torch.tensor([[0.0, 0.0, CAM_DISTANCE]]).to(device)

# 渲染器设置
raster_settings = RasterizationSettings(
    image_size=RENDER_SIZE,
    blur_radius=0.0,
    faces_per_pixel=1,
)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster_settings),
    shader=SoftSilhouetteShader(),
)

# 查找所有中间网格文件
mesh_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "mesh_iter_*.obj")))

# 渲染并合成视频
writer = imageio.get_writer(VIDEO_NAME, fps=FPS)
for fname in tqdm(mesh_files, desc="Rendering frames"):
    try:
        verts, faces, _ = load_obj(fname)
        mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])
        sil = renderer(mesh)[0, ..., 3].cpu().numpy()
        img = (sil * 255).astype(np.uint8)
        writer.append_data(img)
    except Exception as e:
        print(f"Error processing {fname}: {e}")

writer.close()
print(f"Video saved to {VIDEO_NAME}")