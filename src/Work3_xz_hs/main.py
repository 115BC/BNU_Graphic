# main.py
import taichi as ti
from .physics import *
from .config import *

def run():
    # 设置窗口大小
    width, height = 640, 480
    
    # 初始化着色器参数（从中间开始）
    ka[None] = 0.5
    kd[None] = 0.5
    ks[None] = 0.5
    shininess[None] = 64.5
    
    # 创建图像数组
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    
    # 渲染一次初始画面
    render(pixels)
    
    # 创建画布
    gui = ti.GUI("Work3_xz_hs", res=(width, height))
    
    # 显示初始画面
    gui.set_image(pixels)
    
    # 添加滑动条（先创建，后设置初始值）
    slider_ka = gui.slider("Ka (Ambient)", 0.0, 1.0)
    slider_kd = gui.slider("Kd (Diffuse)", 0.0, 1.0)
    slider_ks = gui.slider("Ks (Specular)", 0.0, 1.0)
    slider_shininess = gui.slider("N (Shininess)", 1.0, 128.0)
    
    # 设置滑动条初始值
    slider_ka.value = ka[None]
    slider_kd.value = kd[None]
    slider_ks.value = ks[None]
    slider_shininess.value = shininess[None]
    
    while gui.running:
        # 获取滑动条的值
        ka[None] = slider_ka.value
        kd[None] = slider_kd.value
        ks[None] = slider_ks.value
        shininess[None] = slider_shininess.value
        
        # 渲染所有像素
        render(pixels)
        
        # 显示画面
        gui.set_image(pixels)
        
        # 显示画面
        gui.show()

if __name__ == "__main__":
    run()