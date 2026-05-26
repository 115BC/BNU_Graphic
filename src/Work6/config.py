#d:\code\spring2026\graphic\cg-lib\src\Work6\config.py
import taichi as ti

# 物理参数配置
dt = 0.01  # 时间步长
gravity = -9.8
mass = 0.05
ks = 100.0   # 弹簧劲度系数（平衡重力，产生适度下垂）
kd = 1     # 阻尼系数（增加以抑制震荡）

# 布料网格参数
cloth_size = (20, 20)  # 网格大小
cloth_width = 2.0
cloth_height = 2.0

# 初始化参数
fixed_indices = [(0, 0), (0, cloth_size[1]-1)]  # 固定点索引

# 渲染参数
window_res = (1024, 768)
background_color = 0xAAAAAA
particle_color = 0x3333FF
spring_color = 0x222222
particle_radius = 0.02

# 积分方法枚举
INTEGRATION_METHODS = {
    'explicit': 'Explicit Euler',
    'semi_implicit': 'Semi-Implicit Euler', 
    'implicit': 'Implicit Euler'
}
