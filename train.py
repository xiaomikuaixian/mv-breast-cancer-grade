import argparse
from pathlib import Path
from pprint import pprint
import gc
import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from train_utils.torch import TorchTrainer, TorchLogger
from train_utils.torch.utils import get_time, seed_everything, fit_state_dict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR

from configs import *
from utils import print_config
from metrics import Pfbeta

import torchvision.utils as vutils

class SaveInputImagesCallback:
    def __init__(self, save_dir, interval=10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval  # 每interval个batch保存一次

    def __call__(self, trainer):
        # trainer.batch 是当前batch索引
        if trainer.batch % self.interval == 0:
            images = trainer.input[0].cpu().detach()
            grid = vutils.make_grid(images, normalize=True, scale_each=True)
            save_path = self.save_dir / f'fold{trainer.serial}_epoch{trainer.epoch}_batch{trainer.batch}.png'
            vutils.save_image(grid, save_path)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='GradeClassifierConfig', help="配置文件名称")
    parser.add_argument("--only_fold", type=int, default=-1, help="指定仅训练单个折，默认为全部折")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--debug", action='store_true', help="启用调试模式")
    parser.add_argument("--silent", action='store_true', help="禁用日志文件")
    parser.add_argument("--progress_bar", action='store_true', help="显示训练进度条")
    parser.add_argument("--skip_existing", action='store_true', help="跳过已有的训练折")
    parser.add_argument("--resume", action='store_true', help="从断点继续训练")
    opt = parser.parse_args()
    pprint(opt)

    # 加载配置
    cfg = eval(opt.config)
    export_dir = Path('results') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)

    # 初始化日志记录器
    LOGGER = TorchLogger(export_dir / f'{cfg.name}_{get_time("%y%m%d%H%M")}.log', file=not opt.silent)

    # 设置随机种子并打印配置信息
    seed_everything(cfg.seed, cfg.deterministic)
    print_config(cfg, LOGGER)

    # 加载数据
    train = pd.read_csv(cfg.train_path)
    if opt.debug:
        train = train.iloc[:100]

    # 创建交叉验证索引
    fold_iter = list(cfg.splitter.split(train, train[cfg.target_cols], train[cfg.group_col]))

    # 开始交叉验证训练
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):
        if opt.only_fold >= 0 and fold != opt.only_fold:
            continue

        LOGGER(f'===== 训练折数 {fold} =====')
        train_fold = train.iloc[train_idx]
        valid_fold = train.iloc[valid_idx]

        # 构建训练和验证数据集
        train_data = cfg.dataset(train_fold, cfg.image_dir, cfg.preprocess['train'], cfg.transforms['train'], is_test=False, **cfg.dataset_params)
        valid_data = cfg.dataset(valid_fold, cfg.image_dir, cfg.preprocess['test'], cfg.transforms['test'], is_test=True, **cfg.dataset_params)

        train_loader = D.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        valid_loader = D.DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

        # 构建模型
        model = cfg.model(**cfg.model_params)

        # 加载预训练权重
        if cfg.weight_path:
            weight = torch.load(cfg.weight_path, 'cpu')
            fit_state_dict(weight, model)
            model.load_state_dict(weight, strict=False)
            del weight; gc.collect()

        # 设置分层学习率
        head_lr = cfg.optimizer_params.get('lr', 8e-5)
        backbone_lr = head_lr * 0.2  # 主干网络学习率更低
        classifier_params = list(model.head.parameters())
        backbone_params = [p for i, stage in enumerate(model.encoder.stages) if i >= cfg.model_params['freeze_until'] for p in stage.parameters()]

        # 构建优化器
        optimizer = cfg.optimizer([{'params': backbone_params, 'lr': backbone_lr}, {'params': classifier_params, 'lr': head_lr}], weight_decay=cfg.optimizer_params['weight_decay'])

        # 学习率调度器（Warmup + Cosine）
        scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=3)
        scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[3])

        # 定义训练参数
        FIT_PARAMS = dict(loader=train_loader, loader_valid=valid_loader, criterion=cfg.criterion, optimizer=optimizer,
                          scheduler=scheduler, num_epochs=cfg.num_epochs, callbacks=deepcopy(cfg.callbacks) + [SaveInputImagesCallback(export_dir / 'debug_images', interval=10)],  # 新增保存输入图片的回调,
                          hook=cfg.hook,export_dir=export_dir, eval_metric=cfg.eval_metric, monitor_metrics=cfg.monitor_metrics,
                          fp16=cfg.amp, parallel=cfg.parallel, deterministic=cfg.deterministic, max_grad_norm=cfg.max_grad_norm,
                          grad_accumulations=cfg.grad_accumulations, random_state=cfg.seed, logger=LOGGER,
                          progress_bar=opt.progress_bar, resume=opt.resume)

        # 开始训练
        trainer = TorchTrainer(model, serial=f'fold{fold}', device=None)
        trainer.fit(**FIT_PARAMS)
        torch.cuda.empty_cache()

    # 开始模型推理与评估
    scores, outoffolds, thresholds = [], [], []
    eval_metric = Pfbeta(binarize=True, return_thres=True)

    for fold, (_, valid_idx) in enumerate(fold_iter):
        valid_fold = train.iloc[valid_idx]
        valid_data = cfg.dataset(valid_fold, cfg.image_dir, cfg.preprocess['test'], cfg.transforms['test'], is_test=True, **cfg.dataset_params)
        valid_loader = D.DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

        model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', map_location='cpu')
        fit_state_dict(checkpoint['model'], model)
        model.load_state_dict(checkpoint['model'], strict=False)

        trainer = TorchTrainer(model, serial=f'fold{fold}', device=None)
        preds = trainer.predict(valid_loader)
        targets = valid_data.get_labels()

        # 输出详细的评估指标
        metric_names = ['Pfbeta_raw', 'Pfbeta_binary', 'AUC', 'Accuracy', 'Recall', 'Precision']
        for metric_name, metric_f in zip(metric_names, cfg.monitor_metrics):
            metric_value = metric_f(torch.from_numpy(preds), torch.from_numpy(targets))
            LOGGER(f'{metric_name}: {metric_value:.5f}')

        # 输出PFbeta指标及阈值
        score, thres = eval_metric(torch.from_numpy(preds), torch.from_numpy(targets))
        LOGGER(f'折数{fold} PFbeta: {score:.5f}, 阈值: {thres:.5f}')

        scores.append(score)
        thresholds.append(thres)
        outoffolds.append(preds)

    # 汇总交叉验证评估结果
    LOGGER(f'平均PFbeta: {np.mean(scores):.5f}, 标准差: {np.std(scores):.5f}')