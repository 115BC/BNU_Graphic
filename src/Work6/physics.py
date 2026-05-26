import taichi as ti
from .config import *

# 尝试使用GPU，如果不可用则使用CPU
try:
    ti.init(arch=ti.cuda)
except Exception:
    print("CUDA不可用，使用CPU后端")
    ti.init(arch=ti.cpu)

# 定义向量类型
vec3 = ti.math.vec3

# 粒子数量
n_particles = cloth_size[0] * cloth_size[1]

# 预计算弹簧数量（仅结构弹簧：水平 + 垂直，无对角交叉）
horizontal_springs = cloth_size[0] * (cloth_size[1] - 1)
vertical_springs = (cloth_size[0] - 1) * cloth_size[1]
n_springs = horizontal_springs + vertical_springs

# 粒子属性
x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)       # 位置
v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)       # 速度
f = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)       # 受力

# 弹簧属性
spring_indices = ti.Vector.field(2, dtype=ti.i32, shape=n_springs)
spring_lengths = ti.field(dtype=ti.f32, shape=n_springs)

# 渲染索引
render_lines = ti.field(dtype=ti.i32, shape=(n_springs, 2))

# 最大速度限制（防爆）
max_velocity = 5.0

# 隐式积分迭代中的临时变量
x_new = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
v_new = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
f_new = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)

# 固定点标志与原始位置
is_fixed = ti.field(dtype=ti.i32, shape=n_particles)
original_positions = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)


@ti.func
def compute_forces_on(p_idx):
    """
    计算单个粒子上的重力和阻尼力（@ti.func，会被内联到 kernel 中）
    f_damp = -k_d * v       （阻尼力与速度方向相反）
    f_grav = (0, m*g, 0)   （重力，Y 轴向下）
    """
    force = vec3(0.0)
    # 重力
    force = vec3(0.0, gravity * mass, 0.0)
    # 阻尼力
    force += -kd * v[p_idx]
    return force


@ti.func
def compute_forces_on_with_state(p_idx, pos_field, vel_field):
    """
    使用指定状态场计算单个粒子的重力和阻尼力（用于隐式积分迭代）
    """
    force = vec3(0.0)
    force = vec3(0.0, gravity * mass, 0.0)
    force += -kd * vel_field[p_idx]
    return force


@ti.func
def clamp_velocity(vel):
    """
    限制质点的最大速度，防止在显式欧拉等不稳定方法中出现严重的数值爆炸
    """
    vel_norm = vel.norm()
    if vel_norm > max_velocity:
        vel = vel * (max_velocity / vel_norm)
    return vel


# =============================================================================
#  初始化 Kernel（拆分为多个以保障 GPU 同步）
# =============================================================================

@ti.kernel
def init_cloth():
    """初始化布料网格：粒子位置、速度、弹簧拓扑（仅结构弹簧）"""
    n_rows = cloth_size[0]
    n_cols = cloth_size[1]

    # 1. 粒子位置（网格状排列在 XY 平面）
    for i, j in ti.ndrange(n_rows, n_cols):
        idx = i * n_cols + j
        pos_x = (j / (n_cols - 1) - 0.5) * cloth_width
        pos_y = (1.0 - i / (n_rows - 1)) * cloth_height
        x[idx] = vec3(pos_x, pos_y, 0.0)
        original_positions[idx] = vec3(pos_x, pos_y, 0.0)
        v[idx] = vec3(0.0)
        f[idx] = vec3(0.0)
        is_fixed[idx] = 0

    # 2. 水平弹簧（同行相邻粒子）: o--o--o
    h_per_row = n_cols - 1
    for i, j in ti.ndrange(n_rows, h_per_row):
        s_idx = i * h_per_row + j
        idx_a = i * n_cols + j
        idx_b = i * n_cols + (j + 1)
        spring_indices[s_idx] = ti.Vector([idx_a, idx_b], dt=ti.i32)
        spring_lengths[s_idx] = (x[idx_a] - x[idx_b]).norm()

    # 3. 垂直弹簧（同列相邻粒子）
    v_offset = n_rows * h_per_row
    for i, j in ti.ndrange(n_rows - 1, n_cols):
        s_idx = v_offset + i * n_cols + j
        idx_a = i * n_cols + j
        idx_b = (i + 1) * n_cols + j
        spring_indices[s_idx] = ti.Vector([idx_a, idx_b], dt=ti.i32)
        spring_lengths[s_idx] = (x[idx_a] - x[idx_b]).norm()

    # 4. 固定点（上边两个角）
    fixed_idx1 = 0 * n_cols + 0
    is_fixed[fixed_idx1] = 1

    fixed_idx2 = 0 * n_cols + (n_cols - 1)
    is_fixed[fixed_idx2] = 1


@ti.kernel
def init_render_lines():
    """初始化渲染线条索引"""
    for i in range(n_springs):
        render_lines[i, 0] = spring_indices[i][0]
        render_lines[i, 1] = spring_indices[i][1]


@ti.kernel
def reset_simulation():
    """重置模拟状态"""
    n_rows = cloth_size[0]
    n_cols = cloth_size[1]
    for i, j in ti.ndrange(n_rows, n_cols):
        idx = i * n_cols + j
        pos_x = (j / (n_cols - 1) - 0.5) * cloth_width
        pos_y = (1.0 - i / (n_rows - 1)) * cloth_height
        x[idx] = vec3(pos_x, pos_y, 0.0)
        original_positions[idx] = vec3(pos_x, pos_y, 0.0)
        v[idx] = vec3(0.0)
        f[idx] = vec3(0.0)

    fixed_idx1 = 0 * n_cols + 0
    is_fixed[fixed_idx1] = 1
    fixed_idx2 = 0 * n_cols + (n_cols - 1)
    is_fixed[fixed_idx2] = 1


# =============================================================================
#  力学计算辅助函数（@ti.func，会被 kernel 内联）
# =============================================================================

@ti.func
def _accumulate_spring_forces(force_field, pos_field):
    """
    遍历所有弹簧，使用胡克定律计算弹力并用 atomic_add 累加到力场中。
    胡克定律：f_a = -k_s * (|x_a - x_b| - l0) * (x_a - x_b) / |x_a - x_b|
    其中 (x_a - x_b) / |x_a - x_b| 是从 b 指向 a 的单位方向向量。
    - 拉伸时 (|x| > l0)：力把 a 拉向 b（方向从 a 指向 b = -direction）
    - 压缩时 (|x| < l0)：力把 a 推离 b（方向从 b 指向 a = +direction）
    """
    for s_idx in range(n_springs):
        ia = spring_indices[s_idx][0]
        ib = spring_indices[s_idx][1]
        delta = pos_field[ia] - pos_field[ib]   # 从 ib 指向 ia
        dist = delta.norm()
        l0 = spring_lengths[s_idx]
        if dist > 1e-8:
            # direction: 从 ib → ia 的单位向量
            direction = delta / dist
            # 胡克定律：f_spring_on_a = -k_s * (dist - l0) * direction
            # 拉伸 (dist > l0): f = -k*(+) * dir，负方向 = 指向 ib ✓
            # 压缩 (dist < l0): f = -k*(-) * dir = +k*... * dir，指向 ia ✓
            spring_f = -ks * (dist - l0) * direction
            ti.atomic_add(force_field[ia], spring_f)
            ti.atomic_add(force_field[ib], -spring_f)  # 牛顿第三定律


# =============================================================================
#  三个积分求解器 Kernel（受力计算 + 位置/速度更新合并在同一 Kernel）
# =============================================================================

@ti.kernel
def step_explicit():
    """
    显式欧拉积分
      x_{t+1} = x_t + v_t * dt        （用旧速度更新位置）
      v_{t+1} = v_t + a_t * dt        （用旧加速度更新速度）
    """
    for i in range(n_particles):
        f[i] = vec3(0.0)
    for i in range(n_particles):
        f[i] += compute_forces_on(i)
    _accumulate_spring_forces(f, x)
    for i in range(n_particles):
        if is_fixed[i] == 0:
            old_v = v[i]
            v[i] += (f[i] / mass) * dt
            x[i] += old_v * dt
            v[i] = clamp_velocity(v[i])
        else:
            x[i] = original_positions[i]
            v[i] = vec3(0.0)


@ti.kernel
def step_semi_implicit():
    """
    半隐式欧拉积分（辛欧拉）
      v_{t+1} = v_t + a_t * dt
      x_{t+1} = x_t + v_{t+1} * dt
    """
    for i in range(n_particles):
        f[i] = vec3(0.0)
    for i in range(n_particles):
        f[i] += compute_forces_on(i)
    _accumulate_spring_forces(f, x)
    for i in range(n_particles):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            x[i] += v[i] * dt
            v[i] = clamp_velocity(v[i])
        else:
            x[i] = original_positions[i]
            v[i] = vec3(0.0)


@ti.kernel
def step_implicit_iter():
    """
    隐式欧拉积分（定点迭代法近似求解）
      v_{t+1} = v_t + a_{t+1} * dt
      x_{t+1} = x_t + v_{t+1} * dt
    """
    for i in range(n_particles):
        x_new[i] = x[i]
        v_new[i] = v[i]
    for _ in range(5):
        for i in range(n_particles):
            f_new[i] = vec3(0.0)
        for i in range(n_particles):
            if is_fixed[i] == 0:
                f_new[i] += compute_forces_on_with_state(i, x_new, v_new)
        _accumulate_spring_forces(f_new, x_new)
        for i in range(n_particles):
            if is_fixed[i] == 0:
                v_new[i] = v[i] + (f_new[i] / mass) * dt
                x_new[i] = x[i] + v_new[i] * dt
            else:
                x_new[i] = original_positions[i]
                v_new[i] = vec3(0.0)
    for i in range(n_particles):
        x[i] = x_new[i]
        v[i] = v_new[i]
        if is_fixed[i] == 0:
            v[i] = clamp_velocity(v[i])
        else:
            x[i] = original_positions[i]
            v[i] = vec3(0.0)
