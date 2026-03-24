import taichi as ti
from .config import *
from .physics import *
ti.init(arch=ti.gpu)


def run():
    global v0, v1, v2
    gui=ti.GUI("Work1",res=res)
    z_angle=0.0
    y_angle=0.0
    while gui.running:
        gui.clear()
        
        while gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                z_angle += 1.0  # 增加旋转角度
            elif gui.event.key == 'd':
                z_angle -= 1.0  # 减少旋转角度
            elif gui.event.key == 'w':
                y_angle += 1.0  # 增加旋转角度
            elif gui.event.key == 's':
                y_angle -= 1.0  # 减少旋转角度
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        
        M=get_model_matrix(y_angle,z_angle)
        V=get_view_matrix(eye_position)
        P=get_projection_matrix(60,1,0.1,50)
        MVP=P@V@M
        screen_point=[]
        for i in point:
            v_tmp=ti.Vector([i[0],i[1],i[2],1.0])
            v_tmp=MVP @ v_tmp
            v_tmp = v_tmp / v_tmp[3] if v_tmp[3] != 0 else v_tmp
            v_screen = (v_tmp[0:2] + 1.0) * 0.5  # 映射到 [0, 1] 范围
            screen_point.append(v_screen)
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
            gui.line(begin, end, color=color, radius=2)  # radius 控制线条粗细
        gui.show()
        #print(f"v0_screen: {v0_screen}, v1_screen: {v1_screen}, v2_screen: {v2_screen}")

if __name__ == "__main__":
    run()