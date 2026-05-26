#d:\code\spring2026\graphic\cg-lib\src\Work6\main.py
import taichi as ti
import numpy as np
from .physics import *
from .config import *

def main():
    # 初始化布料
    init_cloth()
    init_render_lines()
    
    # 创建窗口和画布
    window = ti.ui.Window("Mass-Spring Cloth Simulation", window_res, vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    # 设置相机初始位置（布料在 XY 平面，相机在 Z 轴正方向观察）
    camera.position(0, 1.0, 3)
    camera.lookat(0, 1.0, 0)
    
    # 控制变量
    paused = False
    current_method = 'semi_implicit'  # 默认使用半隐式欧拉
    
    while window.running:
        # 执行物理步进（如果未暂停）
        if not paused:
            if current_method == 'explicit':
                step_explicit()
            elif current_method == 'semi_implicit':
                step_semi_implicit()
            elif current_method == 'implicit':
                step_implicit_iter()
        
        # 渲染
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        scene.ambient_light((0.2, 0.2, 0.2))
        scene.point_light(pos=(0.5, 1, 0.5), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1, -0.5), color=(1, 1, 1))
        
        # 绘制布料粒子
        scene.particles(x, radius=particle_radius)
        
        # 绘制弹簧连接
        scene.lines(vertices=x, indices=render_lines, width=1.0)
        
        canvas.scene(scene)
        
        # GUI控制面板
        gui = window.get_gui()
        gui.begin("Simulation Control", 0.05, 0.05, 0.3, 0.25)
        gui.text("Mass-Spring Cloth Simulation")
        paused = gui.checkbox("Pause", paused)
        
        # 积分方法选择
        methods = list(INTEGRATION_METHODS.keys())
        method_names = list(INTEGRATION_METHODS.values())
        current_idx = methods.index(current_method)
        selected_idx = gui.slider_int("Integration Method", current_idx, 0, len(methods)-1)
        current_method = methods[selected_idx]
        gui.text(f"Current method: {method_names[selected_idx]}")
        
        # 重置按钮
        if gui.button("Reset Cloth"):
            reset_simulation()
            
        gui.end()
        
        window.show()

if __name__ == '__main__':
    main()
