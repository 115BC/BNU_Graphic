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

def run():
    # 创建GUI窗口
    window = ti.ui.Window("Work2", res=res)
    canvas = window.get_canvas()
    scene = window.get_scene()  # 使用正确的方式获取Scene对象
    camera = ti.ui.Camera()
    
    # 控制点列表
    control_points = []
    
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
        
        # 计算与绘制逻辑
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