## 基于 Opencv-Python 实现的帧差算法

> 某天在网上冲浪的时候看到这篇文章 [【OpenCV】“帧差法”实现移动物体的检测（车辆识别）](https://blog.csdn.net/wmcy123/article/details/124263832)
> ，感觉很有意思，就浅学了一下 [Opencv](http://www.woshicver.com/) 😜

### 1. 项目目录

```sh
|-- 根目录
	|-- images                  # 图片目录
	    |-- 1.jpg               # 示例图片
	|-- config.py               # 配置文件
	|-- gradient_map.py         # 三种梯度图处理函数
	|-- main.py                 # 主函数
	|-- utils.py                # 工具类
	|-- requirements.txt        # 依赖包列表
	|-- README.md               # 项目介绍 
```

### 2. 环境配置

+ 配置 `Python3.10` 及以上环境

+ 安装依赖包

  ```sh
  pip install -r requirements.txt
  ```

### 3. 使用

+ 传入两张图片
    + 修改 `./config.py` 文件中的 第一张图片路径（`FIRST_PIC_INPUT`）
    + 修改 `./config.py` 文件中的 第二张图片路径（`SECOND_PIC_INPUT`）

+ 自定义阈值
    + 修改 `./config.py` 文件中的 `DIFF_THRESH`
+ 运行主函数，`main.py`

### 4. cv2 内置方法报黄、无提示解决方法

- 确保已经安装 opencv 包

  ```sh
  pip install opencv-python
  ```

- 找到虚拟环境所在位置, 点击加载文件列表

  ![](https://s3.bmp.ovh/imgs/2022/07/14/0e9d5a2681a55629.png)

- 将 `cv2` 文件夹加到 加载文件中

  ![](https://s3.bmp.ovh/imgs/2022/07/14/9ac79229a0d0ead2.png)
