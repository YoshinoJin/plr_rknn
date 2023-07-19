# PLR_rknn

## 简介

本项目在RK3588上部署车牌识别任务，将为分为三个工程部分

使用yolov5进行车牌检测-Plate_Detection_yolov5

使用PLRNet进行车牌识别-Plate_Recognition_lprnet

使用RKNN-toolkit进行模型转换推理部署-PLR_rknn

## 准备工作

### 创建conda虚拟环境

```
conda create --name rknn python=3.8
```

### 激活环境

```
conda activate rknn
```

下载仓库源文件，本项目的yolo模型基于yolov5的[956be8e642b5c10af4a1533e09084ca32ff4f21f](https://github.com/ultralytics/yolov5.git)版本

### 安装依赖环境

先安装依赖包

```
pip install -r requirements-cpu-ubuntu20.04_py38.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple 
```

其次安装rknn_toolkit2，可以在rknn官网找到对应版本的python whl文件。

### 模型转换

对yolov5的onnx与lpr的onnx进行转换。

```
python yolo_2_rknn.py
python plr_2_rknn.py
```

## 验证模型

推理

```
python detect_rknn.py
```

最终得到检测结果无误后便可以部署在rk3588中了。（python）

## TODO

使用C++进行npu的部署加速

