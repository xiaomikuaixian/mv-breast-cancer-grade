from pprint import pformat
import types
import pandas as pd


def print_config(cfg, logger=None):

    def _print(text):
        if logger is None:
            print(text)
        else:
            logger(text)
    
    items = [
        'name', 
        'cv', 'num_epochs', 'batch_size', 'seed', 'train_path', 'addon_train_path', 'image_dir', 
        'dataset', 'dataset_params', 'sampler', 'oversample_ntimes', 'num_classes', 'preprocess', 'transforms', 'splitter',
        'model', 'model_params', 'weight_path', 'optimizer', 'optimizer_params',
        'scheduler', 'scheduler_params', 'batch_scheduler', 'scheduler_target',
        'criterion', 'eval_metric', 'monitor_metrics',
        'amp', 'parallel', 'hook', 'callbacks', 'deterministic', 
        'clip_grad', 'max_grad_norm',
        'pseudo_labels'
    ]
    _print('===== CONFIG =====')
    for key in items:
        try:
            val = getattr(cfg, key)
            if isinstance(val, (type, types.FunctionType)):
                val = val.__name__ + '(*)'
            if isinstance(val, (dict, list)):
                val = '\n'+pformat(val, compact=True, indent=2)
            _print(f'{key}: {val}')
        except:
            _print(f'{key}: ERROR')
    _print(f'===== CONFIGEND =====')

def oversample_data(df, n_times=0):
    def add_oversample_id(df, oid):
        df['oversample_id'] = oid
        return df
    if n_times > 0:
        df = pd.concat([add_oversample_id(df, 0)] + [
            add_oversample_id(df.query('grade_2_categ == 1'), i+1) for i in range(n_times)], axis=0)
    return df


def notify_me(text):
    # line_notify_token = ''
    # line_notify_api = 'https://notify-api.line.me/api/notify'
    # headers = {'Authorization': f'Bearer {line_notify_token}'}
    # data = {'message': '\n' + text}
    # requests.post(line_notify_api, headers=headers, data=data)
    pass