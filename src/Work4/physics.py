# d:\code\spring2026\graphic\cg-lib\src\Work4\physics.py
import taichi as ti
from .config import *

ti.init(arch=ti.gpu)

# 光源位置参数（用于UI交互）
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())

# 最大弹射次数参数（用于UI交互）
max_bounces = ti.field(ti.i32, shape=())

# 初始化默认值
light_pos_x[None] = LIGHT_POS_DEFAULT[0]
light_pos_y[None] = LIGHT_POS_DEFAULT[1]
light_pos_z[None] = LIGHT_POS_DEFAULT[2]
max_bounces[None] = MAX_BOUNCES_DEFAULT

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """计算光线与球体的交点"""
    oc = ro - center
    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4 * a * c
    
    t = -1.0  # 默认值
    
    # 计算交点
    if discriminant >= 0:
        t1 = (-b - ti.sqrt(discriminant)) / (2.0 * a)
        if t1 > 1e-4:
            t = t1
        else:
            t2 = (-b + ti.sqrt(discriminant)) / (2.0 * a)
            if t2 > 1e-4:
                t = t2
    
    return t

@ti.func
def intersect_plane(ro, rd, plane_y):
    """计算光线与平面的交点"""
    t = -1.0  # 默认值
    
    if abs(rd.y) >= 1e-6:  # 光线不平行于平面
        t_calc = (plane_y - ro.y) / rd.y
        if t_calc > 1e-4:  # 避免自相交
            t = t_calc
    
    return t

@ti.func
def get_sphere_normal(point, center):
    """获取球体表面法向量"""
    return (point - center).normalized()

@ti.func
def get_plane_normal():
    """获取平面法向量（向上）"""
    return ti.Vector([0.0, 1.0, 0.0])

@ti.func
def get_material_type(hit_type, obj_id):
    """获取材质类型"""
    material = MATERIAL_DIFFUSE  # 默认值
    
    if hit_type == 0:  # 平面
        material = MATERIAL_DIFFUSE
    elif hit_type == 1:  # 红色球
        material = MATERIAL_DIFFUSE
    elif hit_type == 2:  # 银色球
        material = MATERIAL_MIRROR
    
    return material

@ti.func
def get_material_color(hit_type, obj_id, hit_point):
    """获取材质颜色"""
    color = ti.Vector([0.5, 0.5, 0.5])  # 默认颜色
    
    if hit_type == 0:  # 棋盘格地面
        # 棋盘格纹理
        x_idx = int(ti.floor(hit_point.x * 0.5))
        z_idx = int(ti.floor(hit_point.z * 0.5))
        checker = (x_idx + z_idx) % 2
        if checker:
            color = ti.Vector([0.8, 0.8, 0.8])  # 白色
        else:
            color = ti.Vector([0.2, 0.2, 0.2])  # 黑色
    elif hit_type == 1:  # 红色球
        color = ti.Vector([0.8, 0.1, 0.1])
    elif hit_type == 2:  # 银色球
        color = ti.Vector([0.7, 0.7, 0.7])
    
    return color

@ti.func
def is_in_shadow(point, light_pos):
    """检测点是否在阴影中"""
    shadow = False  # 默认不在阴影中
    
    light_dir = (light_pos - point).normalized()
    light_dist = (light_pos - point).norm()
    
    # 从交点向光源发射阴影射线
    ray_start = point + get_plane_normal() * 1e-4  # 避免自相交 (shadow acne)
    ray_dir = light_dir
    
    # 检查是否有物体阻挡到光源的路径
    closest_t = light_dist  # 光源距离
    
    # 检查与红色球的交点
    t = intersect_sphere(ray_start, ray_dir, 
                        ti.Vector(RED_SPHERE_CENTER), RED_SPHERE_RADIUS)
    if t > 0 and t < closest_t:
        shadow = True
    
    # 检查与银色球的交点
    t = intersect_sphere(ray_start, ray_dir, 
                        ti.Vector(SILVER_SPHERE_CENTER), SILVER_SPHERE_RADIUS)
    if t > 0 and t < closest_t:
        shadow = True
    
    # 检查与地面的交点
    t = intersect_plane(ray_start, ray_dir, GROUND_Y)
    if t > 0 and t < closest_t:
        shadow = True
    
    return shadow

@ti.func
def compute_lighting(hit_point, hit_normal, view_dir, material_color, light_pos):
    """计算光照"""
    # 光源方向
    light_dir = (light_pos - hit_point).normalized()
    light_distance = (light_pos - hit_point).norm()
    
    # 漫反射
    diffuse_intensity = max(0.0, hit_normal.dot(light_dir))
    
    # 环境光
    ambient = 0.2
    
    # 检查阴影
    in_shadow = is_in_shadow(hit_point, light_pos)
    
    lighting = ambient
    if not in_shadow:
        lighting += diffuse_intensity
    
    return material_color * lighting

@ti.kernel
def render(pixels: ti.template(), width: ti.i32, height: ti.i32):
    # 获取当前光源位置
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    max_bounces_local = max_bounces[None]
    
    for i, j in pixels:
        # 将屏幕坐标转换为标准化设备坐标
        u = (i + 0.5) / width
        v = (j + 0.5) / height
        
        # 转换到[-1, 1]区间
        x = 2 * u - 1
        y = 2 * v - 1
        
        # 正确应用宽高比 - 这是关键修正
        aspect_ratio = float(width) / float(height)
        
        # 调整x和y坐标以保持球体的圆形外观
        x *= aspect_ratio
        # 根据需要调整视野
        fov_scale = 1.0  # 可以调整这个值来改变视野
        x *= fov_scale
        y *= fov_scale
        
        # 定义摄像机 - 使用更高的Y位置
        camera_pos = ti.Vector(CAMERA_POS)
        # 构建光线方向 - 从摄像机看向场景
        ray_dir = ti.Vector([x, y, -1.0]).normalized()
        
        # 初始化光线参数
        ray_origin = camera_pos
        throughput = ti.Vector([1.0, 1.0, 1.0])  # 光线吞吐量/衰减系数
        final_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 迭代光线弹射（Whitted-Style Ray Tracing）
        for bounce in range(max_bounces_local):
            closest_t = 1e9
            hit_type = -1  # 0: 平面, 1: 红球, 2: 银球
            hit_obj_id = -1
            hit_point = ti.Vector([0.0, 0.0, 0.0])
            hit_normal = ti.Vector([0.0, 0.0, 0.0])
            
            # 检测与地面的交点
            t = intersect_plane(ray_origin, ray_dir, GROUND_Y)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_type = 0
                hit_obj_id = 0
                hit_point = ray_origin + t * ray_dir
                hit_normal = get_plane_normal()
            
            # 检测与红色球的交点
            t = intersect_sphere(ray_origin, ray_dir, 
                                ti.Vector(RED_SPHERE_CENTER), RED_SPHERE_RADIUS)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_type = 1
                hit_obj_id = 1
                hit_point = ray_origin + t * ray_dir
                hit_normal = get_sphere_normal(hit_point, ti.Vector(RED_SPHERE_CENTER))
            
            # 检测与银色球的交点
            t = intersect_sphere(ray_origin, ray_dir, 
                                ti.Vector(SILVER_SPHERE_CENTER), SILVER_SPHERE_RADIUS)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_type = 2
                hit_obj_id = 2
                hit_point = ray_origin + t * ray_dir
                hit_normal = get_sphere_normal(hit_point, ti.Vector(SILVER_SPHERE_CENTER))
            
            if hit_type == -1:
                # 没有击中任何物体，返回背景色
                # 天空颜色渐变
                sky_color = ti.Vector([0.1, 0.2, 0.5]) * (1.0 - y * 0.5) + ti.Vector([0.3, 0.6, 1.0]) * (y * 0.5)
                final_color += throughput * sky_color
                break
            
            # 获取交点信息
            material_type = get_material_type(hit_type, hit_obj_id)
            material_color = get_material_color(hit_type, hit_obj_id, hit_point)
            
            if material_type == MATERIAL_DIFFUSE:
                # 漫反射材质，计算光照
                view_dir = (-ray_dir).normalized()
                lighting = compute_lighting(hit_point, hit_normal, view_dir, material_color, light_pos)
                final_color += throughput * lighting
                
                # 漫反射材质，停止弹射
                break
            elif material_type == MATERIAL_MIRROR:
                # 镜面反射
                # 使用公式 R = L_in - 2(L_in · N)N
                incident = ray_dir
                reflection = incident - 2.0 * incident.dot(hit_normal) * hit_normal
                
                # 更新光线参数
                ray_origin = hit_point + hit_normal * 1e-4  # 避免自相交 (shadow acne)
                ray_dir = reflection
                
                # 更新光线吞吐量（镜面反射率）
                throughput *= ti.Vector([0.8, 0.8, 0.8])
        
        # 设置像素颜色，限制在[0,1]范围内
        pixels[i, j] = ti.Vector([
            min(1.0, final_color.x),
            min(1.0, final_color.y), 
            min(1.0, final_color.z)
        ])