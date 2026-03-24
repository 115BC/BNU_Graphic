import taichi as ti
from .config import *
from .physics import *
ti.init(arch=ti.gpu)


def run():
    global v0, v1, v2
    gui=ti.GUI("Work1",res=res)
    z_angle=0.0
    while gui.running:
        gui.clear()
        
        while gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                z_angle += 1.0  # 增加旋转角度
            elif gui.event.key == 'd':
                z_angle -= 1.0  # 减少旋转角度
        
        M=get_model_matrix(z_angle)
        V=get_view_matrix(eye_position)
        P=get_projection_matrix(60,1,0.1,50)
        MVP=P@V@M
        v0_tmp = ti.Vector([v0[0], v0[1], v0[2], 1.0])
        v0_tmp = MVP @ v0_tmp
        v0_tmp = v0_tmp / v0_tmp[3] if v0_tmp[3] != 0 else v0_tmp
        # 添加坐标映射：从 [-1, 1] 映射到 [0, 1]
        v0_screen = (v0_tmp[0:2] + 1.0) * 0.5  # 映射到 [0, 1] 范围

        v1_tmp = ti.Vector([v1[0], v1[1], v1[2], 1.0])
        v1_tmp = MVP @ v1_tmp
        v1_tmp = v1_tmp / v1_tmp[3] if v1_tmp[3] != 0 else v1_tmp
        v1_screen = (v1_tmp[0:2] + 1.0) * 0.5  # 映射到 [0, 1] 范围
        v2_tmp = ti.Vector([v2[0], v2[1], v2[2], 1.0])
        v2_tmp = MVP @ v2_tmp
        v2_tmp = v2_tmp / v2_tmp[3] if v2_tmp[3] != 0 else v2_tmp
        v2_screen = (v2_tmp[0:2] + 1.0) * 0.5  # 映射到 [0, 1] 范围

        lines=[(v0_screen,v1_screen,0xFF0000),
           (v0_screen,v2_screen,0x00FF00),
           (v1_screen,v2_screen,0x0000FF)
           ]
        for begin, end, color in lines:
            gui.line(begin, end, color=color, radius=2)  # radius 控制线条粗细
        gui.show()
        #print(f"v0_screen: {v0_screen}, v1_screen: {v1_screen}, v2_screen: {v2_screen}")

if __name__ == "__main__":
    run()