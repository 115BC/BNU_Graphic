import taichi as ti

# 新的中心位置
cx, cy, cz = 2.0, 3.0, 4.0

# 计算新的顶点坐标（原坐标 + 中心偏移量）
v0 = ti.Vector([1.0 + cx, 0.0 + cy, -1.0 + cz])
v1 = ti.Vector([0.0 + cx, 1.0 + cy, -1.0 + cz])
v2 = ti.Vector([-1.0 + cx, 0.0 + cy, -1.0 + cz])
v3 = ti.Vector([0.0 + cx, -1.0 + cy, -1.0 + cz])
v4 = ti.Vector([1.0 + cx, 0.0 + cy, 1.0 + cz])
v5 = ti.Vector([0.0 + cx, 1.0 + cy, 1.0 + cz])
v6 = ti.Vector([-1.0 + cx, 0.0 + cy, 1.0 + cz])
v7 = ti.Vector([0.0 + cx, -1.0 + cy, 1.0 + cz])

point = [v0, v1, v2, v3, v4, v5, v6, v7]

# 相机位置需要相应调整，确保能看到立方体
eye_position = ti.Vector([cx, cy + 20.0, cz ])  # 相机在立方体正前方
cam_center = ti.Vector([cx, cy, cz])           # 相机看向立方体中心
up_vector = ti.Vector([0.0, 0.0, 1.0])
res = (700, 700)

# 旋转矩阵保持不变
R0 = ti.Matrix([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0]
])
R1 = ti.Matrix([
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0]
])