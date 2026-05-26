# d:\code\spring2026\graphic\cg-lib\src\Work4\config.py
import taichi as ti

# 屏幕分辨率
RESOLUTION = (800, 600)

# 摄像机位置 - 提高Y坐标以获得更高视角
CAMERA_POS = (0.0, 2.0, 5.0)

# 光源位置（可变）
LIGHT_POS_DEFAULT = [2.0, 5.0, 3.0]

# 最大弹射次数（可变）
MAX_BOUNCES_DEFAULT = 3

# 材质类型常量
MATERIAL_DIFFUSE = 0
MATERIAL_MIRROR = 1

# 几何体定义
GROUND_Y = -1.0  # 地面Y坐标
RED_SPHERE_CENTER = (-1.5, 0.0, 0.0)
RED_SPHERE_RADIUS = 1.0
SILVER_SPHERE_CENTER = (1.5, 0.0, 0.0)
SILVER_SPHERE_RADIUS = 1.0