import taichi as ti
from .physics import *
from .config import *
ti.init(arch=ti.gpu)

def run():
    gui = ti.GUI("Work1_cx", res=res)
    rotate = 0.0
    while gui.running:
        gui.clear()
        while gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                rotate += 5.0
            elif gui.event.key == 'd':
                rotate -= 5.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 获取正交旋转矩阵
        R = get_R(rotate)
        
        # 计算视图和投影矩阵
        V = get_view_matrix(eye_position)
        P = get_projection_matrix(60, 1, 0.1, 50)
        VP = P @ V
        
        # 应用变换
        screen_point = []
        for p in point:
            # 应用旋转
            rotated_point = R @ p
            # 转换为齐次坐标并应用 VP 变换
            v_tmp = ti.Vector([rotated_point[0], rotated_point[1], rotated_point[2], 1.0])
            v_tmp = VP @ v_tmp
            v_tmp = v_tmp / v_tmp[3] if v_tmp[3] != 0 else v_tmp
            # 映射到屏幕坐标
            v_screen = (v_tmp[0:2] + 1.0) * 0.5
            screen_point.append(v_screen)
        
        # 绘制立方体
        lines = [
            (screen_point[0], screen_point[1], 0xFF0000),
            (screen_point[1], screen_point[2], 0xFF8000),
            (screen_point[2], screen_point[3], 0xFFFF00),
            (screen_point[3], screen_point[0], 0x80FF00),
            (screen_point[4], screen_point[5], 0x00FF00),
            (screen_point[5], screen_point[6], 0x00FF80),
            (screen_point[6], screen_point[7], 0x00FFFF),
            (screen_point[7], screen_point[4], 0x0080FF),
            (screen_point[0], screen_point[4], 0x0000FF),
            (screen_point[1], screen_point[5], 0x8000FF),
            (screen_point[2], screen_point[6], 0xFF00FF),
            (screen_point[3], screen_point[7], 0xFF0080)
        ]
        
        for begin, end, color in lines:
            gui.line(begin, end, color=color, radius=2)
        
        gui.show()
'''
def run():
    gui = ti.GUI("Work1_cx", res=res)
    rotate = 0.0
    while gui.running:
        gui.clear()
        while gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                rotate += 5.0  # 增加旋转角度
            elif gui.event.key == 'd':
                rotate -= 5.0  # 减少旋转角度
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 计算视图和投影矩阵
        V = get_view_matrix(eye_position)
        P = get_projection_matrix(60, 1, 0.1, 50)
        VP = P @ V
        ''''''
        # 显示多个过渡状态的立方体
        num_transitions = 5  # 过渡状态数量
        for i in range(num_transitions):
            # 计算当前过渡状态的角度
            transition_angle = rotate + i * 36  # 每个状态间隔 36 度
            # 获取当前过渡状态的旋转矩阵
            R = get_R(transition_angle)
            
            # 应用变换到所有顶点
            screen_point = []
            for p in point:
                # 应用旋转
                rotated_point = R @ p
                # 转换为齐次坐标并应用 VP 变换
                v_tmp = ti.Vector([rotated_point[0], rotated_point[1], rotated_point[2], 1.0])
                v_tmp = VP @ v_tmp
                v_tmp = v_tmp / v_tmp[3] if v_tmp[3] != 0 else v_tmp
                # 映射到屏幕坐标
                v_screen = (v_tmp[0:2] + 1.0) * 0.5
                screen_point.append(v_screen)
            
            # 绘制当前过渡状态的立方体
            lines = [
                (screen_point[0], screen_point[1], 0xFF0000),  # 底面边1 - 红色
                (screen_point[1], screen_point[2], 0xFF8000),  # 底面边2 - 橙色
                (screen_point[2], screen_point[3], 0xFFFF00),  # 底面边3 - 黄色
                (screen_point[3], screen_point[0], 0x80FF00),  # 底面边4 - 黄绿色
                (screen_point[4], screen_point[5], 0x00FF00),  # 顶面边1 - 绿色
                (screen_point[5], screen_point[6], 0x00FF80),  # 顶面边2 - 青绿色
                (screen_point[6], screen_point[7], 0x00FFFF),  # 顶面边3 - 青色
                (screen_point[7], screen_point[4], 0x0080FF),  # 顶面边4 - 天蓝色
                (screen_point[0], screen_point[4], 0x0000FF),  # 侧棱1 - 蓝色
                (screen_point[1], screen_point[5], 0x8000FF),  # 侧棱2 - 靛蓝色
                (screen_point[2], screen_point[6], 0xFF00FF),  # 侧棱3 - 紫色
                (screen_point[3], screen_point[7], 0xFF0080)   # 侧棱4 - 粉红色
            ]
            
            for begin, end, color in lines:
                gui.line(begin, end, color=color, radius=2)
        
        gui.show()
'''
if __name__ == "__main__":
    run()