import taichi as ti
from .config import *
ti.init(arch=ti.gpu)
# 定义缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=1001)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        # 从curve_points_field中读取浮点坐标
        point = curve_points_field[i]
        # 将其映射为整数像素索引（乘以宽高）
        x = int(point[0] * 800)
        y = int(point[1] * 800)
        # 进行0 <= x < 800和0 <= y < 800的越界检查
        if x >= 0 and x < 800 and y >= 0 and y < 800:
            # 给pixels缓冲区对应的位置赋绿色
            pixels[x, y] = ti.Vector([0.0, 1.0, 0.0])  # 绿色（使用正确的访问语法）