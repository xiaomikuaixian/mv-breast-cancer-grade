import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import pydicom
from general import DATA_DIR
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut


IMG_SIZE = 2048
WINDOW = True  
VOI_LUT = True
VOI_LUT_SIGONLY = True
print(DATA_DIR)
EXPORT_DIR = DATA_DIR/f'image_resized_{IMG_SIZE}{"V" if VOI_LUT else ""}{"S" if VOI_LUT_SIGONLY else ""}'
EXPORT_DIR.mkdir(exist_ok=True)
N_JOBS = 4


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def process(f, size=1024):
    patient_id = f.parent.name
    if not (EXPORT_DIR / patient_id).exists():
        (EXPORT_DIR / patient_id).mkdir(exist_ok=True)
    image_id = f.stem
    dicom = pydicom.dcmread(f)

    img = dicom.pixel_array

    if VOI_LUT:
        if dicom.get('VOILUTFunction', None) != 'SIGMOID' and VOI_LUT_SIGONLY:
            img = img  # 不做VOI LUT
        else:
            img = apply_voi_lut(img, dicom)

    if WINDOW:
        # 尝试获取窗位和窗宽，如果不存在则使用图像的最大最小值
        window_center = dicom.get('WindowCenter', img.min() + (img.max() - img.min()) / 2)
        window_width = dicom.get('WindowWidth', img.max() - img.min())

        # 确保窗宽和窗位是数值 (有些DICOM文件可能有多个值，取第一个)
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = window_center[0]
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = window_width[0]

        # 如果获取失败,使用最大最小值
        if window_center is None:
          window_center = img.min() + (img.max() - img.min()) / 2
        if window_width is None:
          window_width = img.max() - img.min()

        img_min = window_center - window_width / 2
        img_max = window_center + window_width / 2
        img = np.clip(img, img_min, img_max)  # 使用 NumPy 的 clip 函数更高效
        img = (img - img_min) / (img_max - img_min)

    if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img

    img = cv2.resize(img, (size, size))
    cv2.imwrite(str(EXPORT_DIR / f'{patient_id}/{image_id}.png'), (img * 255).astype(np.uint8))

if __name__ == '__main__':
    
    # step1:读取数据集,转成png格式
    train_images = list((DATA_DIR / 'train_images/').glob('**/*.DCM'))
    print(f'{len(train_images)}条数据')
    _ = ProgressParallel(n_jobs=N_JOBS)(
        delayed(process)(img_path, size=IMG_SIZE) for img_path in tqdm(train_images)
    )