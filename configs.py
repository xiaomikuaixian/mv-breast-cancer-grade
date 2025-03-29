from pathlib import Path
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from train_utils.torch.callbacks import (
    EarlyStopping,SaveAverageSnapshot, CollectTopK)
from train_utils.torch.hooks import TrainHook
from train_utils.metrics import AUC
from metrics import Accuracy, Recall, Precision

from general import *
from datasets import PatientLevelDataset
from metrics import PRAUC, Pfbeta
from transforms import AutoFlip, CropROI
from architectures import MultiViewModel

class GradeClassifierConfig:
    name = 'grade_classifier'
    seed = 2025
    train_path = DATA_DIR / 'train_images/train.csv'
    addon_train_path = None
    image_dir = Path('input/BC_MG/image_resized_2048')

    cv = 5
    splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    target_cols = ['grade_2_categ']
    group_col = 'patient_id'
    dataset = PatientLevelDataset
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path=None
    )

    model = MultiViewModel
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        freeze_layers=True,
        freeze_until=4
    )
    # weight_path = 'pretrained_models/pretrained_convnext_2048.pth.tar'
    weight_path = None

    num_epochs = 30
    batch_size = 12
    optimizer = optim.AdamW
    optimizer_params = dict(lr=8e-5, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    scheduler_target = None
    batch_scheduler = False

    criterion = nn.BCEWithLogitsLoss()  #
    eval_metric = PRAUC().torch
    monitor_metrics = [
        Pfbeta(binarize=False),
        Pfbeta(binarize=True),
        AUC().torch,
        Accuracy().torch,
        Recall().torch,
        Precision().torch,
    ]

    amp = True
    parallel = None
    deterministic = False
    clip_grad = None
    max_grad_norm = 100
    grad_accumulations = 1
    hook = TrainHook()

    callbacks = [
        CollectTopK(3, maximize=True),
        SaveAverageSnapshot(num_snapshot=3),
        EarlyStopping(patience=6, maximize=True, skip_epoch=2)
    ]

    preprocess = dict(
        train=A.Compose([
            CropROI(buffer=80),
            AutoFlip(sample_width=100),
            A.Resize(1024, 512)  # 1536×768，捕获更多细节
        ], bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([
            CropROI(buffer=80),
            AutoFlip(sample_width=200),
            A.Resize(1024, 512)  # 1536×768，捕获更多细节
        ], bbox_params=A.BboxParams(format='pascal_voc')),
    )

    transforms = dict(
    train=A.Compose([

        A.Lambda(image=lambda x, **kwargs: x.astype(np.float32) / 255.0),

        # 平移、缩放和旋转变换
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20,
                           border_mode=cv2.BORDER_CONSTANT, value=0),
        # 水平翻转,概率0.5
        A.HorizontalFlip(p=0.5),

        # 垂直翻转,概率0.2
        A.VerticalFlip(p=0.2),

        # 随机调整亮度和对比度,概率0.5
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),

        # 随机模糊,概率0.25
        A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.25),

        # 自适应直方图均衡化,概率0.1
        A.CLAHE(p=0.1), 

        # 以下三种变形中随机选择一种,概率0.1
        A.OneOf([
            # 弹性变形
            A.ElasticTransform(alpha=60, sigma=60*0.05, alpha_affine=60*0.03),
            # 网格扭曲
            A.GridDistortion(),
            # 光学畸变
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.1),
        ], p=0.1),
        
        # 标准化
        A.Normalize(mean=0.485, std=0.229, max_pixel_value=1.0, always_apply=True),

        # 随机遮挡,最多20个96x96的方块,概率0.2
        A.CoarseDropout(max_holes=20, max_height=96, max_width=96, p=0.2)
    ]),
    test=A.Compose([
        A.Lambda(image=lambda x, **kwargs: x.astype(np.float32) / 255.0),
        # 标准化
        A.Normalize(mean=0.485, std=0.229, max_pixel_value=1.0, always_apply=True),
    ])
)

    pseudo_labels = None
    debug = False