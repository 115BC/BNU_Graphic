# d:\code\spring2026\graphic\cg-lib\src\Work4_xz_jc\main.py
import taichi as ti
from .physics import *
from .config import *

def run():
    # 设置窗口大小
    width, height = RESOLUTION
    
    # 创建图像数组
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    
    # 创建窗口
    gui = ti.GUI("Work4_xz_jc - Anti-Aliasing", res=(width, height))
    
    # 添加滑动条控件
    slider_light_x = gui.slider("Light X", -5.0, 5.0)
    slider_light_y = gui.slider("Light Y", 0.0, 10.0)
    slider_light_z = gui.slider("Light Z", -5.0, 5.0)
    slider_max_bounces = gui.slider("Max Bounces", 1, 5, step=1)
    
    # 设置滑动条初始值
    slider_light_x.value = light_pos_x[None]
    slider_light_y.value = light_pos_y[None]
    slider_light_z.value = light_pos_z[None]
    slider_max_bounces.value = max_bounces[None]
    
    # 跟踪滑动条值的变化
    prev_light_x = light_pos_x[None]
    prev_light_y = light_pos_y[None]
    prev_light_z = light_pos_z[None]
    prev_max_bounces = max_bounces[None]
    
    while gui.running:
        # 获取滑动条的当前值
        current_light_x = slider_light_x.value
        current_light_y = slider_light_y.value
        current_light_z = slider_light_z.value
        current_max_bounces = int(slider_max_bounces.value)
        
        # 只有当值发生变化时才更新
        if (current_light_x != prev_light_x or 
            current_light_y != prev_light_y or 
            current_light_z != prev_light_z or 
            current_max_bounces != prev_max_bounces):
            
            light_pos_x[None] = current_light_x
            light_pos_y[None] = current_light_y
            light_pos_z[None] = current_light_z
            max_bounces[None] = current_max_bounces
            
            # 更新前一个值
            prev_light_x = current_light_x
            prev_light_y = current_light_y
            prev_light_z = current_light_z
            prev_max_bounces = current_max_bounces
        
        # 渲染所有像素
        render(pixels, width, height)
        
        # 显示画面
        gui.set_image(pixels)
        
        # 显示画面
        gui.show()

if __name__ == "__main__":
    run()