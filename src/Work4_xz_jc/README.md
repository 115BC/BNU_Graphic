# 实验五:光线追踪
- 理论理解： 理解光线投射（Ray Casting）与光线追踪（Ray Tracing）的本质区别。
- 全局光照： 掌握如何通过发射“次级射线（Secondary Rays）”来实现硬阴影（Hard Shadows）和理想镜面反射（Perfect Reflection）。
- GPU 编程思维： 学习如何将传统的“递归”光线追踪算法改写为适合 GPU 并行计算的“迭代（循环）”模式。
- 抗锯齿 (Anti-Aliasing, MSAA) (+10%)： 目前物体边缘有明显的像素锯齿。在每个像素内随机采样多次（发射多条主射线），并将颜色平均，实现平滑的边缘过渡。
![实验五](../../img/work4_xz_jc.gif)
