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
        # 计算精确的浮点像素坐标
        float_x = point[0] * 800.0
        float_y = point[1] * 800.0
        
        # 考察3x3网格的像素
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                # 计算当前像素的整数坐标
                pixel_x = int(float_x) + dx
                pixel_y = int(float_y) + dy
                
                # 进行边界检查
                if pixel_x >= 0 and pixel_x < 800 and pixel_y >= 0 and pixel_y < 800:
                    # 计算像素中心点的坐标（像素中心在整数坐标+0.5处）
                    center_x = float(pixel_x) + 0.5
                    center_y = float(pixel_y) + 0.5
                    
                    # 计算像素中心点与精确浮点坐标的距离
                    distance = ti.sqrt((center_x - float_x) ** 2 + (center_y - float_y) ** 2)
                    
                    # 基于距离计算颜色权重（距离越近，权重越大）
                    # 使用高斯函数或线性衰减模型
                    max_distance = ti.sqrt(2.0)  # 3x3网格中最远的距离
                    weight = 1.0 - distance / max_distance
                    weight = ti.max(0.0, weight)  # 确保权重非负
                    
                    # 应用权重到颜色
                    green_value = weight * 1.0
                    
                    # 混合颜色到像素（使用加法混合）
                    current_color = pixels[pixel_x, pixel_y]
                    new_color = ti.Vector([0.0, green_value, 0.0])
                    # 简单的加法混合，避免颜色过饱和
                    pixels[pixel_x, pixel_y] = ti.min(current_color + new_color, ti.Vector([1.0, 1.0, 1.0]))