# d:\code\spring2026\graphic\cg-lib\src\Work4_xz_bl\physics.py
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

    t = -1.0

    if discriminant >= 0:
        sqrt_disc = ti.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        if t1 > 1e-4:
            t = t1
        else:
            t2 = (-b + sqrt_disc) / (2.0 * a)
            if t2 > 1e-4:
                t = t2

    return t


@ti.func
def intersect_plane(ro, rd, plane_y):
    """计算光线与平面的交点"""
    t = -1.0

    if abs(rd.y) >= 1e-6:
        t_calc = (plane_y - ro.y) / rd.y
        if t_calc > 1e-4:
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
    material = MATERIAL_DIFFUSE

    if hit_type == 0:  # 平面
        material = MATERIAL_DIFFUSE
    elif hit_type == 1:  # 玻璃球
        material = MATERIAL_GLASS
    elif hit_type == 2:  # 银色球
        material = MATERIAL_MIRROR

    return material


@ti.func
def get_material_color(hit_type, obj_id, hit_point):
    """获取材质颜色"""
    color = ti.Vector([0.5, 0.5, 0.5])

    if hit_type == 0:  # 棋盘格地面
        x_idx = int(ti.floor(hit_point.x * 0.5))
        z_idx = int(ti.floor(hit_point.z * 0.5))
        checker = (x_idx + z_idx) % 2
        if checker:
            color = ti.Vector([0.8, 0.8, 0.8])  # 白色
        else:
            color = ti.Vector([0.2, 0.2, 0.2])  # 黑色
    elif hit_type == 1:  # 玻璃球（红色调）
        color = ti.Vector([0.9, 0.1, 0.1])
    elif hit_type == 2:  # 银色球
        color = ti.Vector([0.7, 0.7, 0.7])

    return color


@ti.func
def is_in_shadow(point, hit_normal, light_pos):
    """检测点是否在阴影中。
    
    关键：使用传入的 hit_normal 来偏移阴影射线起点，避免 Shadow Acne。
    """
    shadow = False

    light_dir = (light_pos - point).normalized()
    light_dist = (light_pos - point).norm()

    # ★ 核心避坑：使用交点处的法线进行偏移，而非硬编码平面法线
    eps = 1e-4
    ray_start = point + hit_normal * eps
    ray_dir = light_dir

    closest_t = light_dist

    # 玻璃球不产生阴影（透明材质，光线可穿透）

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
    light_dir = (light_pos - hit_point).normalized()

    # 漫反射
    diffuse_intensity = max(0.0, hit_normal.dot(light_dir))

    # 环境光
    ambient = 0.2

    # 检查阴影 - 传入 hit_normal 用于 Shadow Acne 修复
    in_shadow = is_in_shadow(hit_point, hit_normal, light_pos)

    lighting = ambient
    if not in_shadow:
        lighting += diffuse_intensity

    return material_color * lighting


@ti.kernel
def render(pixels: ti.template(), width: ti.i32, height: ti.i32):
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    max_bounces_local = max_bounces[None]

    for i, j in pixels:
        u = (i + 0.5) / width
        v = (j + 0.5) / height

        x = 2 * u - 1
        y = 2 * v - 1

        aspect_ratio = float(width) / float(height)
        x *= aspect_ratio

        camera_pos = ti.Vector(CAMERA_POS)
        ray_dir = ti.Vector([x, y, -1.0]).normalized()

        ray_origin = camera_pos
        throughput = ti.Vector([1.0, 1.0, 1.0])
        final_color = ti.Vector([0.0, 0.0, 0.0])
        current_refractive_index = 1.0  # 当前介质的折射率（1.0 = 空气）

        for bounce in range(max_bounces_local):
            closest_t = 1e9
            hit_type = -1
            hit_point = ti.Vector([0.0, 0.0, 0.0])
            hit_normal = ti.Vector([0.0, 0.0, 0.0])

            # 检测与地面的交点
            t = intersect_plane(ray_origin, ray_dir, GROUND_Y)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_type = 0
                hit_point = ray_origin + t * ray_dir
                hit_normal = get_plane_normal()

            # 检测与玻璃球的交点
            glass_center = ti.Vector(GLASS_SPHERE_CENTER)
            t = intersect_sphere(ray_origin, ray_dir, glass_center, GLASS_SPHERE_RADIUS)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_type = 1
                hit_point = ray_origin + t * ray_dir
                hit_normal = get_sphere_normal(hit_point, glass_center)

            # 检测与银色镜面球的交点
            silver_center = ti.Vector(SILVER_SPHERE_CENTER)
            t = intersect_sphere(ray_origin, ray_dir, silver_center, SILVER_SPHERE_RADIUS)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_type = 2
                hit_point = ray_origin + t * ray_dir
                hit_normal = get_sphere_normal(hit_point, silver_center)

            if hit_type == -1:
                # 没有击中任何物体，返回背景色
                sky_color = ti.Vector([0.1, 0.2, 0.5])
                final_color += throughput * sky_color
                break

            material_type = get_material_type(hit_type, 0)
            material_color = get_material_color(hit_type, 0, hit_point)

            if material_type == MATERIAL_DIFFUSE:
                # 漫反射材质，计算光照后停止弹射
                view_dir = (-ray_dir).normalized()
                lighting = compute_lighting(hit_point, hit_normal, view_dir, material_color, light_pos)
                final_color += throughput * lighting
                break

            elif material_type == MATERIAL_MIRROR:
                # 镜面反射
                incident = ray_dir
                # 反射公式: R = L_in - 2 * (L_in · N) * N
                reflection = incident - 2.0 * incident.dot(hit_normal) * hit_normal

                # ★ 沿法线偏移避免自相交 (Shadow Acne)
                eps = 1e-4
                ray_origin = hit_point + hit_normal * eps
                ray_dir = reflection

                # ★ 更新光线吞吐量（镜面反射率）
                throughput *= ti.Vector([MIRROR_REFLECTANCE, MIRROR_REFLECTANCE, MIRROR_REFLECTANCE])

            elif material_type == MATERIAL_GLASS:
                # 玻璃材质：使用斯涅尔定律计算折射
                incident = ray_dir
                normal = hit_normal

                # 判断是进入还是离开玻璃球
                dot = incident.dot(normal)
                entering = dot < 0

                # 计算折射率比值 (Snell's Law: n1 * sin(theta1) = n2 * sin(theta2))
                # eta = n1 / n2, where n1 = current medium, n2 = target medium
                eta = 1.0  # 默认值
                if entering:
                    # 从空气进入玻璃: eta = 1.0 / 1.5
                    eta = 1.0 / GLASS_REFRACT_INDEX
                else:
                    # 从玻璃进入空气: eta = 1.5 / 1.0
                    eta = GLASS_REFRACT_INDEX / 1.0
                    normal = -normal  # 翻转法线，使其指向内部

                # 斯涅尔定律计算
                cos_i = -incident.dot(normal)
                sin_t2 = eta * eta * (1.0 - cos_i * cos_i)

                eps = 1e-4

                if sin_t2 > 1.0:
                    # 全内反射（Total Internal Reflection）
                    reflection = incident - 2.0 * incident.dot(normal) * normal
                    ray_origin = hit_point - normal * eps
                    ray_dir = reflection
                else:
                    # 正常折射
                    cos_t = ti.sqrt(1.0 - sin_t2)
                    refracted = eta * incident + (eta * cos_i - cos_t) * normal

                    ray_origin = hit_point - normal * eps
                    ray_dir = refracted.normalized()

                    # 更新当前折射率
                    if entering:
                        current_refractive_index = GLASS_REFRACT_INDEX
                    else:
                        # 成功离开玻璃，回到空气
                        current_refractive_index = 1.0

                # 玻璃颜色过滤（轻微着色，保持透明感）
                throughput *= material_color * 0.05 + ti.Vector([0.95, 0.95, 0.95])

        # 设置像素颜色，限制在[0,1]范围内
        pixels[i, j] = ti.Vector([
            min(1.0, final_color.x),
            min(1.0, final_color.y),
            min(1.0, final_color.z)
        ])
