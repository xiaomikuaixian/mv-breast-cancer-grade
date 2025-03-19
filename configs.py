from pathlib import Path
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from train_utils.torch.callbacks import (
    EarlyStopping, SaveSnapshot,SaveAverageSnapshot, CollectTopK)
from train_utils.torch.hooks import TrainHook
from train_utils.metrics import AUC

from general import *
from datasets import *
from loss_functions import *
from metrics import *
from transforms import *
from architectures import *
from training_extras import *
from global_objectives.losses import AUCPRLoss


class Baseline:
    name = 'baseline'
    seed = 2025
    train_path = DATA_DIR/'train_images/train.csv'
    addon_train_path = None
    image_dir = Path('input/BC_MG/image_resized_2048')
    cv = 5
    splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    target_cols = ['grade_2_categ']
    group_col = 'patient_id'
    dataset = PatientLevelDataset
    dataset_params = dict()
    sampler = None
    oversample_ntimes = 0

    model = MultiViewModel
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True,
    )
    weight_path = None
    num_epochs = 15
    batch_size = 16
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    scheduler_target = None
    batch_scheduler = False
    criterion = BCEWithLogitsLoss()
    eval_metric = Pfbeta(binarize=True)
    monitor_metrics = [AUC().torch, PRAUC().torch, Pfbeta(binarize=False)]
    amp = True
    parallel = None
    deterministic = False
    clip_grad = None
    max_grad_norm = 100
    grad_accumulations = 1
    hook = TrainHook()
    callbacks = [
        EarlyStopping(patience=6, maximize=True, skip_epoch=0),
        SaveSnapshot()
    ]

    preprocess = dict(
        train=None,
        test=None,
    )

    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30), 
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )

    pseudo_labels = None
    debug = False


class Baseline4(Baseline):
    name = 'baseline_4'
    cv = 5
    seed = 2025
    splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    dataset_params = dict(
        sample_criteria='low_value_for_implant'
    )
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), 
            RandomCropROI(threshold=(0.08, 0.12), buffer=(-20, 100)), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)]),
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.1),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            ToTensorV2()
        ])
    )
    image_dir = Path('input/BC_MG/image_resized_2048')
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True)
    optimizer = optim.AdamW
    optimizer_params = dict(lr=1e-5, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    eval_metric = Pfbeta(average_both=True)
    monitor_metrics = [AUC().torch, PRAUC().torch, Pfbeta(binarize=False),  Pfbeta(binarize=True),]
    parallel = 'ddp'
    callbacks = [
        EarlyStopping(patience=6, maximize=True, skip_epoch=5),
        SaveSnapshot()
    ]


class Aug07(Baseline4):
    name = 'aug_07'
    dataset_params = dict(
        sample_criteria='low_value_for_implant',
        bbox_path='input/BC_MG/bbox_all.csv',
    )
    preprocess = dict(
        train=A.Compose([
            RandomCropBBox(buffer=(-20, 100)), 
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )


class Aug07lr0(Aug07):
    name = 'aug_07_lr0'
    criterion = AUCPRLoss()
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 45, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.25),
            A.CLAHE(p=0.1), 
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.CoarseDropout(max_holes=20, max_height=64, max_width=64, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            ToTensorV2()
        ])
    )
    callbacks = [
        CollectTopK(3, maximize=True), 
        SaveAverageSnapshot(num_snapshot=3)
    ]
    eval_metric = PRAUC().torch
    monitor_metrics = [Pfbeta(binarize=False), Pfbeta(binarize=True)]
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path='input/BC_MG/bbox_all.csv',
    )

# 新增：乳腺癌组织分级优化配置
class GradeClassifierConfig(Aug07lr0):
    name = 'grade_classifier'
    weight_path = 'pretrained_models/pretrained_convnext_2048.pth.tar'
    
    # 使用较小分辨率以减少训练时间但保持性能
    preprocess = dict(
        train=A.Compose([
            CropROI(buffer=80),
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )
    
    # 增加监控指标，全面评估模型性能
    eval_metric = PRAUC().torch
    monitor_metrics = [
        Pfbeta(binarize=False), 
        Pfbeta(binarize=True),
        AUC().torch,  # ROC-AUC
    ]
    
    # 调整优化器参数，使用更高学习率以加速微调
    optimizer_params = dict(lr=5e-5, weight_decay=1e-6)
    
    # 减少训练轮数，配合早停策略
    num_epochs = 15
    
    # 修改回调函数，提前早停，避免过拟合
    callbacks = [
        CollectTopK(3, maximize=True), 
        SaveAverageSnapshot(num_snapshot=3),
        EarlyStopping(patience=4, maximize=True, skip_epoch=2)
    ]
    
    # 冻结早期层，只微调后期层
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        freeze_until=4      # 冻结前4层卷积块
    )
    
    # 对于小数据集，不使用分布式训练
    parallel = None
    
    # 数据集参数调整
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path=None
    )