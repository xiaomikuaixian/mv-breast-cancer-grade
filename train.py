import argparse
from pathlib import Path
from pprint import pprint
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import pickle
import traceback

# 导入Kuma的TorchTrainer和TorchLogger炼丹炉
from train_utils.torch import TorchTrainer, TorchLogger
from train_utils.torch.utils import get_time, seed_everything, fit_state_dict
from train_utils.utils import sigmoid
from timm.layers import convert_sync_batchnorm

# 导入我的配置文件
from configs import *
from utils import print_config, notify_me, oversample_data
# 导入我的指标
from metrics import Pfbeta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--only_fold", type=int, default=-1,
                        help="train only specified fold")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--inference", action='store_true', help="inference")
    parser.add_argument("--extended_inference", action='store_true')
    parser.add_argument("--per_site_inference", action='store_true')
    parser.add_argument("--tta", action='store_true', 
                        help="test time augmentation ")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--skip_existing", action='store_true')
    parser.add_argument("--calibrate", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--wait", type=int, default=0,
                        help="time (sec) to wait before execution")
    opt = parser.parse_args()
    pprint(opt)

    ''' Configure hardware '''
    opt.gpu = None # use all visible GPUs
    
    ''' Configure path '''
    cfg = eval(opt.config)
    export_dir = Path('results') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)
    opt.num_workers = min(cfg.batch_size, opt.num_workers)

    ''' Configure logger '''
    log_items = [
        'epoch', 'train_loss', 'train_metric', 'train_monitor', 
        'valid_loss', 'valid_metric', 'valid_monitor', 
        'learning_rate', 'early_stop'
    ]
    if opt.debug:
        log_items += ['gpu_memory']
    if opt.only_fold >= 0:
        logger_path = f'{cfg.name}_fold{opt.only_fold}_{get_time("%y%m%d%H%M")}.log'
    else:
        logger_path = f'{cfg.name}_{get_time("%y%m%d%H%M")}.log'
    LOGGER = TorchLogger(
        export_dir / logger_path, 
        log_items=log_items, file=not opt.silent
    )
    if opt.wait > 0:
        LOGGER(f'Waiting for {opt.wait} sec.')
        time.sleep(opt.wait)

    ''' Prepare data '''
    seed_everything(cfg.seed, cfg.deterministic)
    print_config(cfg, LOGGER)
    train = pd.read_csv(cfg.train_path)
    # data preprocessor
    if opt.debug:
        train = train.iloc[:1000]
    splitter = cfg.splitter
    fold_iter = list(splitter.split(X=train, y=train[cfg.target_cols], groups=train[cfg.group_col]))
    
    '''
    Training
    '''
    scores = []
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):
        
        if opt.only_fold >= 0 and fold != opt.only_fold:
            continue  # skip fold

        if opt.inference:
            continue

        if opt.skip_existing and (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'checkpoint fold{fold}.pt already exists.')
            continue

        LOGGER(f'===== TRAINING FOLD {fold} =====')

        train_fold = train.iloc[train_idx]
        valid_fold = train.iloc[valid_idx]
        train_fold = oversample_data(train_fold, cfg.oversample_ntimes)

        # dataset 为PatientLevelDataset对象
        train_data = cfg.dataset(
            df=train_fold,
            image_dir=cfg.image_dir,
            preprocess=cfg.preprocess['train'],
            transforms=cfg.transforms['train'],
            is_test=False,
            **cfg.dataset_params)
        valid_data = cfg.dataset(
            df=valid_fold, 
            image_dir=cfg.image_dir,
            preprocess=cfg.preprocess['test'],
            transforms=cfg.transforms['test'],
            is_test=True,
            **cfg.dataset_params)
        if cfg.addon_train_path is not None:
            addon_train = pd.read_csv(cfg.addon_train_path)
            if 'fold' in addon_train.columns:
                addon_train = addon_train.loc[addon_train['fold'] == fold]
            train_data.update_df(addon_train)
        # 统计正负样本数量
        train_weights = train_data.get_labels().reshape(-1)
        valid_weights = valid_data.get_labels().reshape(-1)
        train_weights[train_weights == 1] = (train_weights == 0).sum() / (train_weights == 1).sum()
        train_weights[train_weights == 0] = 1
        if cfg.sampler is not None:
            sampler = cfg.sampler(train_weights.tolist(), len(train_weights))
        else:
            sampler = None
        LOGGER(f'train count: {len(train_data)} / valid count: {len(valid_data)}')
        LOGGER(f'train pos: {train_data.get_labels().reshape(-1).mean():.5f} / valid pos: {valid_weights.mean():.5f}')

        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, 
            shuffle=True if cfg.sampler is None else False,
            sampler=sampler, num_workers=opt.num_workers, pin_memory=True)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True)

        model = cfg.model(**cfg.model_params)

        # Load snapshot
        if cfg.weight_path is not None:
            if cfg.weight_path.is_dir():
                weight_path = cfg.weight_path / f'fold{fold}.pt'
            else:
                weight_path = cfg.weight_path
            LOGGER(f'{weight_path} loaded.')
            weight = torch.load(weight_path, 'cpu')['model']
            fit_state_dict(weight, model)
            model.load_state_dict(weight, strict=False)
            del weight; gc.collect()

        # 在创建优化器前，添加分层学习率设置
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'stages') and 'freeze_until' in cfg.model_params:
            # 区分不同层使用不同学习率
            freeze_until = cfg.model_params.get('freeze_until', 0)
            
            # 分层学习率参数
            head_lr = cfg.optimizer_params.get('lr', 5e-5)  # 分类头使用较高学习率
            backbone_lr = head_lr * 0.4  # 主干网络使用较低学习率
            
            # 参数分组
            classifier_params = list(model.head.parameters())
            backbone_params = []
            
            # 仅包含未冻结的主干网络层
            for i, stage in enumerate(model.encoder.stages):
                if i >= freeze_until:  # 只考虑未冻结的层
                    backbone_params.extend(list(stage.parameters()))
            
            # 创建参数组
            param_groups = [
                {'params': backbone_params, 'lr': backbone_lr},
                {'params': classifier_params, 'lr': head_lr}
            ]
            
            optimizer = cfg.optimizer(param_groups, **{k: v for k, v in cfg.optimizer_params.items() if k != 'lr'})
        else:
            # 默认优化器
            optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)
        scheduler = cfg.scheduler(optimizer, **cfg.scheduler_params)
        FIT_PARAMS = {
            'loader': train_loader,
            'loader_valid': valid_loader,
            'criterion': cfg.criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scheduler_target': cfg.scheduler_target,
            'batch_scheduler': cfg.batch_scheduler, 
            'num_epochs': cfg.num_epochs,
            'callbacks': deepcopy(cfg.callbacks),
            'hook': cfg.hook,
            'export_dir': export_dir,
            'eval_metric': cfg.eval_metric,
            'monitor_metrics': cfg.monitor_metrics,
            'fp16': cfg.amp,
            'parallel': cfg.parallel,
            'deterministic': cfg.deterministic, 
            'clip_grad': cfg.clip_grad, 
            'max_grad_norm': cfg.max_grad_norm,
            'grad_accumulations': cfg.grad_accumulations, 
            'random_state': cfg.seed,
            'logger': LOGGER,
            'progress_bar': opt.progress_bar, 
            'resume': opt.resume
        }
        try:
            trainer = TorchTrainer(model, serial=f'fold{fold}', device=None)
            trainer.ddp_sync_batch_norm = convert_sync_batchnorm
            trainer.ddp_params = dict(
                broadcast_buffers=True, 
                find_unused_parameters=True
            )
            trainer.fit(**FIT_PARAMS)
            
            # 添加：在每个epoch后获取训练和验证指标，监控过拟合情况
            train_metrics = trainer.history['train_monitor']
            valid_metrics = trainer.history['valid_monitor']
            
            if len(train_metrics) > 0 and len(valid_metrics) > 0:
                # 计算最后5个epoch的平均差异
                last_n = min(5, len(train_metrics))
                train_avg = sum(train_metrics[-last_n:]) / last_n
                valid_avg = sum(valid_metrics[-last_n:]) / last_n
                metric_diff = train_avg - valid_avg
                
                LOGGER(f"训练过程监控 - 最后{last_n}个epoch:")
                LOGGER(f"平均训练指标: {train_avg:.4f}, 平均验证指标: {valid_avg:.4f}")
                LOGGER(f"指标差异: {metric_diff:.4f} ({'疑似过拟合' if metric_diff > 0.1 else '正常'})")
                
                # 记录训练和验证损失的差异
                if 'train_loss' in trainer.history and 'valid_loss' in trainer.history:
                    train_loss = trainer.history['train_loss'][-last_n:]
                    valid_loss = trainer.history['valid_loss'][-last_n:]
                    avg_train_loss = sum(train_loss) / last_n
                    avg_valid_loss = sum(valid_loss) / last_n
                    loss_diff = avg_valid_loss - avg_train_loss
                    
                    LOGGER(f"平均训练损失: {avg_train_loss:.4f}, 平均验证损失: {avg_valid_loss:.4f}")
                    LOGGER(f"损失差异: {loss_diff:.4f} ({'疑似过拟合' if loss_diff > 0.1 else '正常'})")
                
        except Exception as e:
            err = traceback.format_exc()
            LOGGER(err)
            if not opt.silent:
                notify_me('\n'.join([
                    f'[{cfg.name}:fold{opt.only_fold}]', 
                    'Training stopped due to:', 
                    f'{traceback.format_exception_only(type(e), e)}'
                ]))
        del model, trainer, train_data, valid_data; gc.collect()
        torch.cuda.empty_cache()


    '''
    Prediction and calibration
    '''
    outoffolds = []
    oof_targets = []
    thresholds = []
    sites = []
    eval_metric = Pfbeta(binarize=True, return_thres=True)
    
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):

        if not (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'fold{fold}.pt missing. No target to predict.')
            continue

        LOGGER(f'===== INFERENCE FOLD {fold} =====')

        valid_fold = train.iloc[valid_idx]
        valid_data = cfg.dataset(
            df=valid_fold, 
            image_dir=cfg.image_dir,
            preprocess=cfg.preprocess['test'],
            transforms=cfg.transforms['test'],
            is_test=True,
            **cfg.dataset_params)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True)

        model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', 'cpu')
        # clean up checkpoint
        if 'checkpoints' in checkpoint.keys():
            del checkpoint['checkpoints']
            torch.save(checkpoint, export_dir/f'fold{fold}.pt')
        fit_state_dict(checkpoint['model'], model)
        model.load_state_dict(checkpoint['model'])
        del checkpoint; gc.collect()
        if cfg.parallel == 'ddp':
            model = convert_sync_batchnorm(model)
            inference_parallel = 'dp'
            valid_loader = D.DataLoader(
                valid_data, batch_size=cfg.batch_size*4, shuffle=False,
                num_workers=opt.num_workers, pin_memory=False)
        else:
            inference_parallel = None

        trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
        trainer.logger = LOGGER
        trainer.register(hook=cfg.hook, callbacks=cfg.callbacks)
        pred_logits = trainer.predict(valid_loader, parallel=inference_parallel, progress_bar=opt.progress_bar)
        target_fold = valid_data.get_labels()
        valid_sites = [valid_data.df_dict[valid_data.pids[i]]['site_id'].values[0] for i in range(len(valid_data))]

        if cfg.hook.__class__.__name__ == 'SingleImageAggregatedTrain': # max aggregation
            valid_fold['prediction'] = pred_logits.reshape(-1)
            agg_df = valid_fold.groupby(['patient_id', 'laterality']).agg(
                {'prediction': 'max', 'grade_2_categ': 'first'})
            pred_logits = agg_df['prediction'].values.reshape(-1, 1)
            target_fold = agg_df['grade_2_categ'].values.reshape(-1, 1)
        
        if opt.extended_inference: # Use logits for aggregation
            results = []
            for (_, pid, lat), p, t, s in zip(valid_data.pids, pred_logits.reshape(-1), target_fold.reshape(-1), valid_sites):
                results.append({'prediction_id': f'{pid}_{lat}', 'pred': p, 'target': t, 'site_id': s})
            results = pd.DataFrame(results).groupby('prediction_id').agg({'pred': 'mean', 'target': 'max', 'site_id': 'max'}).reset_index()
            pred_logits = results['pred'].values.reshape(-1, 1)
            target_fold = results['target'].values.reshape(-1, 1)
            valid_sites = results['site_id'].values.tolist()

        if opt.per_site_inference:
            sites.append(valid_sites)
            results = pd.DataFrame({
                'site_id': valid_sites, 
                'pred': pred_logits.reshape(-1), 
                'pred_bin': pred_logits.reshape(-1), 
                'label': target_fold.reshape(-1)})
            thres_fold = []
            for site in [1, 2]:
                score_site, thres_site = eval_metric(
                    torch.from_numpy(results.query('site_id == @site')['pred'].values),
                    torch.from_numpy(results.query('site_id == @site')['label'].values), )
                LOGGER(f'Site {site} PFbeta: {score_site:.5f} threshold: {thres_site:.5f}')
                results.loc[results['site_id'] == site, 'pred_bin'] = (sigmoid(results.loc[results['site_id'] == site, 'pred']) > thres_site).astype(float)
                thres_fold.append(thres_site)
            pred_logits = results['pred'].values.reshape(-1, 1)
            target_fold = results['label'].values.reshape(-1, 1)
            eval_score_fold = eval_metric.pfbeta(results['label'].values, results['pred_bin'].values)
            LOGGER(f'Overall PFbeta: {eval_score_fold:.5f}')
            thresholds.append(thres_fold)
        else:
            eval_score_fold, thres = eval_metric(torch.from_numpy(pred_logits), torch.from_numpy(target_fold))
            LOGGER(f'PFbeta: {eval_score_fold:.5f} threshold: {thres:.5f}')
            thresholds.append(thres)

        for im, metric_f in enumerate(cfg.monitor_metrics):
            LOGGER(f'Monitor metric {im}: {metric_f(torch.from_numpy(pred_logits), torch.from_numpy(target_fold)):.5f}')
        
        scores.append(eval_score_fold)
        outoffolds.append(pred_logits)
        oof_targets.append(target_fold)
        
        torch.cuda.empty_cache()
    
    with open(str(export_dir/f'predictions{"_ext" if opt.extended_inference else ""}.pickle'), 'wb') as f:
        pickle.dump({
            'folds': fold_iter,
            'outoffolds': outoffolds, 
            'targets': oof_targets,
            'sites': sites,
            'thresholds': thresholds
        }, f)

    LOGGER(f'scores: {scores}')
    LOGGER(f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}')
    if not opt.silent:
        notify_me('\n'.join([
            f'[{cfg.name}:fold{opt.only_fold}]',
            'Training has finished successfully.',
            f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}'
        ]))
