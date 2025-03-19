# 乳腺癌组织学分级自动化系统

## 项目简介

本项目旨在使用深度学习技术自动对乳腺癌图像进行组织学分级。利用卷积神经网络和多视图融合技术，从乳腺钼靶图像中识别并分类恶性肿瘤，提供准确的组织学分级预测。

## 功能特点

- 自动化乳腺钼靶图像预处理与增强
- 多视图（CC、MLO等）的图像分析与融合
- 基于ConvNeXt等先进CNN架构的图像特征提取
- 精确的乳腺癌恶性程度分级
- 支持交叉验证和模型集成

## 项目结构

```
mv-breast-cancer-grade/
├── architectures.py    # 模型架构定义
├── configs.py          # 配置参数
├── convert_image.py    # DCM转PNG图像处理
├── crop.py             # 图像裁剪处理
├── datasets.py         # 数据集加载与预处理
├── general.py          # 通用功能
├── global_objectives/  # 全局目标优化
├── loss_functions.py   # 损失函数定义
├── metrics.py          # 评估指标
├── modules.py          # 模型组件
├── train.py            # 训练脚本
├── train_utils/        # 训练工具库
└── utils.py            # 辅助函数
```


## 安装与准备

1. 克隆此仓库
2. 创建并激活虚拟环境

```bash
conda env create -f environment.yaml
conda activate kumaconda22
```


3. 准备数据

将原始DICOM影像放入`input/BC_MG/train_images/`目录下，运行图像转换脚本：

```bash
python convert_image.py
```


## 使用方法

### 训练模型

```bash
python train.py --config GradeClassifierConfig --num_workers 4
```


### 参数说明

- `--config`: 选择配置文件中定义的配置
- `--only_fold`: 仅训练特定折的数据
- `--num_workers`: 数据加载的工作线程数
- `--debug`: 启用调试模式
- `--inference`: 仅执行推理

## 模型架构

项目使用`MultiViewModel`架构，支持多种CNN骨干网络，如ConvNeXt、EfficientNet等。该架构能够处理多视图乳腺图像，并将不同视图的特征融合以提高分类准确率。

## 数据处理

- 支持DICOM格式转换为PNG
- 自动图像裁剪与对齐
- 多种数据增强：旋转、翻转、对比度调整等
- 支持多视图采样策略

## 评估指标

- PFBeta评分
- ROC-AUC
- PR-AUC

## 依赖项

主要依赖库:
- PyTorch 1.11.0
- timm
- albumentations
- pydicom
- OpenCV
- scikit-learn

完整依赖见`environment.yaml`文件。

## 注意事项

本项目处理的是医学图像数据，请确保您有权限访问和使用相关数据，并遵守隐私和数据保护规定。 