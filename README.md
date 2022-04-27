# Miaooo3D

我自己写的3D引擎


是[计算机图形学入门-闫令琪](https://www.bilibili.com/video/BV1X7411F744)的代码实现

# 截图

![图片](https://github.com/miaooo0000OOOO/My3DEngine_OpencvNumpy/blob/master/screenshot/2022-04-24-18-09-45%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png?raw=true
)


![图片](https://github.com/miaooo0000OOOO/My3DEngine_OpencvNumpy/blob/master/screenshot/2022-04-26-22-16-13%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png?raw=true)

![图片](https://github.com/miaooo0000OOOO/My3DEngine_OpencvNumpy/blob/master/screenshot/2022-04-27-12-21-11%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png?raw=true)

![图片](https://github.com/miaooo0000OOOO/My3DEngine_OpencvNumpy/blob/master/screenshot/2022-04-27-12-22-14%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png?raw=true)

# 环境

python : opencv, numpy

# 运行

python core.py

也可以从fileops.py开始运行

# 使用说明

使用wasdqe控制摄像头平移

wasdqe分别对应前左后右上下

使用ijkluo控制摄像头旋转（我也不知道为啥有误差啊）

ijkluo分别对应仰，向左偏航，俯，向右偏航，向左滚转，向右滚转

f切换线框与体积显示

esc退出

python真的好慢啊啊啊啊(不要用到生产环境中)

# 文件解释

core.py 核心文件 主要代码

matrix.py 生成矩阵函数 主要代码

fileops.py 文件操作 用来导入STL模型 主要代码

iiid.py 旧的代码

ign.py 旧的代码

objects.py 存储三维模型的地方

# 参考

导入STL模型

https://zhuanlan.zhihu.com/p/398208443

STL模型来源

https://clara.io/view/8124b5cc-5584-484f-8622-97d8548b8e4f