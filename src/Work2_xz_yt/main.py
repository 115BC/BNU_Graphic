import taichi as ti
from .physics import *
from .config import *
#ti.init(arch=ti.gpu)
 
def de_casteljau(points, t):
    n = len(points)
    if n == 0:
        return ti.Vector([0.0, 0.0])
    
    # 创建临时数组存储中间点
    temp_points = points.copy()
    
    # 递归计算中间点
    for i in range(1, n):
        for j in range(n - i):
            # 线性插值：P_j^i = (1-t)*P_j^(i-1) + t*P_{j+1}^(i-1)
            temp_points[j] = (1 - t) * temp_points[j] + t * temp_points[j + 1]
    
    return temp_points[0]

def compute_b_spline(points, t):
    """
    计算均匀三次B样条曲线上的点
    
    参数:
        points: 控制点列表，至少需要4个点
        t: 参数值，范围[0, 1]
    
    返回:
        B样条曲线上对应参数t的点
    """
    n = len(points)
    if n < 4:
        return ti.Vector([0.0, 0.0])
    
    # 确定当前t值所在的段
    num_segments = n - 3
    segment_index = min(int(t * num_segments), num_segments - 1)
    local_t = (t * num_segments) - segment_index
    
    # 获取当前段的4个控制点
    p0 = points[segment_index]
    p1 = points[segment_index + 1]
    p2 = points[segment_index + 2]
    p3 = points[segment_index + 3]
    
    # 均匀三次B样条的基矩阵
    # [ -1  3 -3  1 ]
    # [  3 -6  3  0 ]
    # [ -3  0  3  0 ]
    # [  1  4  1  0 ] / 6
    
    # 计算基函数值
    t2 = local_t * local_t
    t3 = t2 * local_t
    
    # 基函数
    N0 = (-t3 + 3*t2 - 3*local_t + 1) / 6.0
    N1 = (3*t3 - 6*t2 + 4) / 6.0
    N2 = (-3*t3 + 3*t2 + 3*local_t + 1) / 6.0
    N3 = t3 / 6.0
    
    # 计算曲线上的点
    point = N0 * p0 + N1 * p1 + N2 * p2 + N3 * p3
    return point

def run():
    # 创建GUI窗口
    window = ti.ui.Window("Work2_xz_yt", res=res)
    canvas = window.get_canvas()
    scene = window.get_scene()  # 使用正确的方式获取Scene对象
    camera = ti.ui.Camera()
    
    # 控制点列表
    control_points = []
    
    # 曲线模式：True为B样条曲线，False为贝塞尔曲线
    b_spline_mode = False
    
    while window.running:
        # 清空pixels缓冲区
        pixels.fill(0)
        
        # 处理鼠标点击事件
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB and len(control_points) < MAX_CONTROL_POINTS:
                # 获取鼠标坐标并归一化到[0, 1]范围
                pos = window.get_cursor_pos()
                control_points.append(ti.Vector([pos[0], pos[1]]))
            # 处理键盘清空事件
            elif window.event.key == 'c':
                control_points.clear()
            # 处理模式切换事件
            elif window.event.key == 'b':
                b_spline_mode = not b_spline_mode
                print(f"切换到{'B样条曲线' if b_spline_mode else '贝塞尔曲线'}模式")
        
        # 计算与绘制逻辑
        if b_spline_mode:
            # B样条曲线模式（需要至少4个控制点）
            if len(control_points) >= 4:
                # 在CPU端循环1000次，算出所有曲线上的点
                curve_points = []
                for i in range(1001):
                    t = i / 1000.0
                    point = compute_b_spline(control_points, t)
                    curve_points.append([point[0], point[1]])
                
                # 使用curve_points_field.from_numpy(...)批量拷贝到GPU
                import numpy as np
                curve_points_np = np.array(curve_points, dtype=np.float32)
                curve_points_field.from_numpy(curve_points_np)
                
                # 调用draw_curve_kernel(1001)让GPU画图
                draw_curve_kernel(1001)
        else:
            # 贝塞尔曲线模式（需要至少2个控制点）
            if len(control_points) >= 2:
                # 在CPU端循环1000次，算出所有曲线上的点
                curve_points = []
                for i in range(1001):
                    t = i / 1000.0
                    point = de_casteljau(control_points, t)
                    curve_points.append([point[0], point[1]])
                
                # 使用curve_points_field.from_numpy(...)批量拷贝到GPU
                import numpy as np
                curve_points_np = np.array(curve_points, dtype=np.float32)
                curve_points_field.from_numpy(curve_points_np)
                
                # 调用draw_curve_kernel(1001)让GPU画图
                draw_curve_kernel(1001)
        
        # 绘制控制点（使用对象池技巧）
        import numpy as np
        # 创建长度为100的NumPy数组，全部填充为-10.0（藏在屏幕外面）
        gui_points_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        # 把真实的控制点覆盖到数组的前几个位置
        for i, point in enumerate(control_points):
            if i < MAX_CONTROL_POINTS:
                gui_points_np[i] = [point[0], point[1]]
        # 将其from_numpy给gui_points
        gui_points.from_numpy(gui_points_np)
        
        # 绘制控制点（放大显示）
        for i in range(MAX_CONTROL_POINTS):
            point = gui_points[i]
            # 检查点是否在屏幕内
            if point[0] >= 0 and point[0] <= 1 and point[1] >= 0 and point[1] <= 1:
                x = int(point[0] * 800)
                y = int(point[1] * 800)
                # 绘制一个3x3的红色方块作为控制点
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx = x + dx
                        ny = y + dy
                        if nx >= 0 and nx < 800 and ny >= 0 and ny < 800:
                            # 绘制红色控制点
                            pixels[nx, ny] = ti.Vector([1.0, 0.0, 0.0])  # 红色
        
        # 绘制控制点之间的绿色线段
        for i in range(len(control_points) - 1):
            # 获取当前点和下一个点
            p1 = control_points[i]
            p2 = control_points[i + 1]
            
            # 转换为像素坐标
            x1 = int(p1[0] * 800)
            y1 = int(p1[1] * 800)
            x2 = int(p2[0] * 800)
            y2 = int(p2[1] * 800)
            
            # 绘制线段（使用Bresenham算法）
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            
            x, y = x1, y1
            while True:
                # 绘制当前点
                if x >= 0 and x < 800 and y >= 0 and y < 800:
                    pixels[x, y] = ti.Vector([0.0, 1.0, 0.0])  # 绿色
                
                # 检查是否到达终点
                if x == x2 and y == y2:
                    break
                
                # 计算下一步
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy
        
        # 将pixels传给canvas.set_image(pixels)显示
        canvas.set_image(pixels)
        window.show()

if __name__ == "__main__":
    run()