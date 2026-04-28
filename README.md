# 黄花菜的计算机图形学代码库

## 项目架构

- BNU_Graphic 
  - img <span style="color:#999">*保存了每一次实验的示例动图*</span>
  - src <span style="color:#999">*项目各个实验的源代码*</span>
    - Work0:第一次实验之装环境与熟悉git用法
    - Work1:实验二的基础任务,学习了MVP变换的基本方法
    - Work1_xz:实验二的选做任务,构造了三维正方体的旋转与映射
    - Work1_xz_cz:实验二的选做任务,利用插值法进行旋转操作
    - Work2:实验三的基础任务,构建了Bézier曲线的渲染
    - Work2_xz_yt:
    - Work2_xz_zy:
    - Work3:
    - Work3_xz_bp:
    - Work3_xz_hs:
    

# 课程实验

## 实验一:图形学开发工具
- 学习了环境搭建与git的用法
- 由于啥都不会，所以实验一的代码来自参考教程
![实验一](img/work0.gif)

## 实验二:旋转与变换
- 学习使用了Taichi的kernel核函数的使用方法,以及其基本语法内容
- 学习理解了MVP变换的大致内容
- 学习并构建二维和三维图形的旋转与映射
- 学习了插值法实现图形的旋转
- PS:三维图形的映射在绘制线的先后顺序还有点问题
![实验二](img/work1.gif)
![实验二选做](img/work1_xz.gif)
![实验二插值](img/work1_cz.gif)

## 实验三:Bézier曲线/B样条曲线
- 理解贝塞尔曲线 (Bézier Curve) 的几何意义。
- 理解并用代码实现计算贝塞尔曲线的 De Casteljau 算法。
- 掌握“光栅化”的基础概念：如何在像素缓冲区 (Frame Buffer) 中直接操作和点亮像素。
- 掌握现代化图形界面中的鼠标点击与交互事件处理。

![实验三曲线](img/work2_qx.gif)
![实验三走样](img/work2_xz_zy.gif)
![实验三样条](img/work2_xz_yt.gif)

# 实验四:Phong光照模型
- 理论理解： 理解并掌握局部光照的基本原理，区分环境光（Ambient）、漫反射（Diffuse）和镜面高光（Specular）。
- 数学基础： 熟练掌握三维空间中的向量运算（法向量计算、光线方向、视线方向与反射向量）。
- 工程实践： 掌握如何利用 Taichi 实现交互式渲染，通过 UI 控件实时调节材质参数，直观感受各个参数对渲染结果的影响。

![实验四](img/work3.gif)
![实验四BP](img/work3_xz_bp.gif)
![实验四HS](img/work3_xz_hs.gif)