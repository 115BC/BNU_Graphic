import taichi as ti
from .config import *

ti.init(arch=ti.cpu)

# 定义着色器参数
ka = ti.field(ti.f32, shape=())
kd = ti.field(ti.f32, shape=())
ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

# 初始化默认值（从中间开始）
ka[None] = 0.5
kd[None] = 0.5
ks[None] = 0.5
shininess[None] = 64.5

@ti.kernel
def render(pixels: ti.template()):
    width, height = pixels.shape[0], pixels.shape[1]
    for i, j in pixels:
        # 将像素坐标转换为标准化设备坐标
        x = (i / width - 0.5) * 2
        y = (j / height - 0.5) * 2 * (height / width)
        
        # 初始化光线
        ray_origin = ti.Vector([camera[0], camera[1], camera[2]])
        # 计算光线方向：从摄像机指向像素，调整缩放因子
        # 移除固定的z=0.0，使用与摄像机相同的z坐标比例
        ray_target = ti.Vector([x * 0.8, y * 0.8, -1.0])  # 使用-1.0作为相对z坐标
        ray_direction = ray_target.normalized()
        
        # 初始化颜色为背景色（深藏蓝色 rgb(0,34,53)）
        color = ti.Vector([0.0, 34/255, 53/255])
        closest_t = 1e9  # 初始化为一个很大的值
        
        # 检测与红色球体的交点
        sphere_center = ti.Vector(red_sphere[0])
        sphere_radius = red_sphere[1]
        sphere_color = ti.Vector(red_sphere[2])
        
        oc = ray_origin - sphere_center
        a = ray_direction.dot(ray_direction)
        b = 2.0 * oc.dot(ray_direction)
        c = oc.dot(oc) - sphere_radius * sphere_radius
        discriminant = b * b - 4 * a * c
        
        if discriminant > 0:
            t = (-b - ti.sqrt(discriminant)) / (2 * a)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_point = ray_origin + t * ray_direction
                normal = (hit_point - sphere_center).normalized()
                # 计算Phong着色
                # 环境光
                ambient = ka[None] * sphere_color
                
                # 漫反射
                light_pos = ti.Vector(light_point[0])
                light_color = ti.Vector(light_point[1])
                light_dir = (light_pos - hit_point).normalized()
                diffuse = kd[None] * sphere_color * light_color * max(0.0, normal.dot(light_dir))
                
                # 镜面高光（Blinn-Phong模型）
                camera_pos = ti.Vector([camera[0], camera[1], camera[2]])
                view_dir = (camera_pos - hit_point).normalized()
                # 计算半程向量H
                half_vector = (view_dir + light_dir).normalized()
                # Blinn-Phong镜面高光
                specular = ks[None] * light_color * pow(max(0.0, normal.dot(half_vector)), shininess[None])
                
                # 总颜色
                color = ambient + diffuse + specular
        
        # 检测与紫色圆锥的交点
        cone_vertex = ti.Vector(purple_cone[0])
        cone_base_y = purple_cone[1]
        cone_base_radius = purple_cone[2]
        cone_color = ti.Vector(purple_cone[3])
        
        # 计算圆锥的高度和斜率
        cone_height = cone_vertex[1] - cone_base_y
        cone_slope = cone_base_radius / cone_height
        
        # 光线与圆锥的交点计算
        A = ray_direction[0]**2 + ray_direction[2]**2 - (cone_slope**2) * ray_direction[1]**2
        B = 2 * (ray_origin[0] * ray_direction[0] + ray_origin[2] * ray_direction[2] - 
                 (cone_slope**2) * (ray_origin[1] - cone_vertex[1]) * ray_direction[1])
        C = ray_origin[0]**2 + ray_origin[2]**2 - (cone_slope**2) * (ray_origin[1] - cone_vertex[1])**2
        
        discriminant_cone = B * B - 4 * A * C
        
        if discriminant_cone > 0:
            t1 = (-B - ti.sqrt(discriminant_cone)) / (2 * A)
            t2 = (-B + ti.sqrt(discriminant_cone)) / (2 * A)
            
            # 选择合适的t值
            t = -1.0
            if t1 > 0:
                hit_point1 = ray_origin + t1 * ray_direction
                if hit_point1[1] >= cone_base_y and hit_point1[1] <= cone_vertex[1]:
                    t = t1
            if t2 > 0 and (t < 0 or t2 < t):
                hit_point2 = ray_origin + t2 * ray_direction
                if hit_point2[1] >= cone_base_y and hit_point2[1] <= cone_vertex[1]:
                    t = t2
            
            if t > 0 and t < closest_t:
                hit_point = ray_origin + t * ray_direction
                # 计算圆锥表面的法线
                dx = 2 * (hit_point[0] - cone_vertex[0])
                dy = -2 * (cone_slope**2) * (hit_point[1] - cone_vertex[1])
                dz = 2 * (hit_point[2] - cone_vertex[2])
                normal = ti.Vector([dx, dy, dz]).normalized()
                # 计算Phong着色
                # 环境光
                ambient = ka[None] * cone_color
                
                # 漫反射
                light_pos = ti.Vector(light_point[0])
                light_color = ti.Vector(light_point[1])
                light_dir = (light_pos - hit_point).normalized()
                diffuse = kd[None] * cone_color * light_color * max(0.0, normal.dot(light_dir))
                
                # 镜面高光（Blinn-Phong模型）
                camera_pos = ti.Vector([camera[0], camera[1], camera[2]])
                view_dir = (camera_pos - hit_point).normalized()
                # 计算半程向量H
                half_vector = (view_dir + light_dir).normalized()
                # Blinn-Phong镜面高光
                specular = ks[None] * light_color * pow(max(0.0, normal.dot(half_vector)), shininess[None])
                
                # 总颜色
                color = ambient + diffuse + specular
        
        # 设置像素颜色
        pixels[i, j] = color