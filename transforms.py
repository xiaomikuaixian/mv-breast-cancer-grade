import numpy as np
import pandas as pd
import random
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform


'''
Preprocessing
'''
def get_blue_ratio(img):
    # (N, C, W, H) image
    rgbs = img.transpose(0, 3, 1, 2).mean(2).mean(2)  # N, C
    br = (100 + rgbs[:, 2]) * 256 / \
        (1 + rgbs[:, 0] + rgbs[:, 1]) / (1 + rgbs.sum(1))
    return br


def make_tiles(img, sz=128, num_tiles=4, criterion='darkness', concat=True, dropout=0.0):
    if len(img.shape) == 2:
        img = img[:, :, None]
    w, h, ch = img.shape
    pad0, pad1 = (sz - w % sz) % sz, (sz - h % sz) % sz
    padding = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]]
    img = np.pad(img, padding, mode='constant', constant_values=0)
    img = img.reshape(img.shape[0]//sz, sz, img.shape[1]//sz, sz, ch)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, ch)
    valid_count = len(img)
    if len(img) < num_tiles:
        padding = [[0, num_tiles-len(img)], [0, 0], [0, 0], [0, 0]]
        img = np.pad(img, padding, mode='constant', constant_values=255)
    if criterion == 'darkness':
        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_tiles]
    elif criterion == 'blue-ratio':
        idxs = np.argsort(get_blue_ratio(img) * -1)[:num_tiles]
    elif criterion == 'brightness':
        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[::-1][:num_tiles]
    else:
        raise ValueError(criterion)
    if concat:
        tile_row = int(np.sqrt(num_tiles))
        img = cv2.hconcat(
            [cv2.vconcat([_img for _img in img[idxs[i:i+tile_row]]]) \
                for i in np.arange(0, num_tiles, tile_row)])
    else:
        img = img[idxs]
        if dropout > 0:
            valid_count = min(valid_count, num_tiles)
            drop_count = round(valid_count * dropout)
            if drop_count > 0:
                drop_index = random.sample(range(valid_count), drop_count)
                img[drop_index] = img[drop_index].mean()
    return img
    


class CropROI(ImageOnlyTransform):
    """基于最大连通组件的ROI裁剪。
    
    Args:
        threshold (float): 二值化阈值，范围[0,1]
        buffer (int): ROI周围的缓冲区大小（像素）
        always_apply (bool): 是否总是应用此变换
        p (float): 应用此变换的概率
    """
    def __init__(self, threshold=0.1, buffer=80, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.threshold = threshold
        self.buffer = buffer

    def apply(self, img, **params):
        # 转换为float32类型
        img = img.astype(np.float32)
        
        # 确保输入图像是灰度图
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        # 1. 图像归一化
        normalized = img_gray.astype(float) / 255.0

        # 2. 二值化
        binary = (normalized > self.threshold).astype(np.uint8)

        # 3. 寻找连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # 4. 找到最大的连通组件（排除背景）
        if num_labels > 1:  # 确保至少有一个连通组件（除背景外）
            # 获取所有连通组件的面积（排除背景）
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_label = np.argmax(areas) + 1  # +1 因为背景标签为0
            
            # 创建最大连通组件的掩码
            mask = (labels == max_label).astype(np.uint8)

            # 5. 添加缓冲区
            if self.buffer > 0:
                kernel = np.ones((self.buffer, self.buffer), np.uint8)
                mask = cv2.dilate(mask, kernel)

            # 6. 获取ROI边界
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if np.any(rows) and np.any(cols):  # 确保找到了非空区域
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]

                # 7. 添加缓冲区（确保不超出图像边界）
                rmin = max(0, rmin)
                rmax = min(img.shape[0], rmax)
                cmin = max(0, cmin)
                cmax = min(img.shape[1], cmax)

                # 8. 裁剪图像
                return img[rmin:rmax, cmin:cmax]

        # 如果没有找到有效的连通组件，返回原始图像
        return img

    def get_transform_init_args_names(self):
        return ("threshold", "buffer")


class RandomCropROI(DualTransform):

    def __init__(self, threshold=(0.08, 0.12), buffer=(0, 160), always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.threshold = threshold
        self.buffer = buffer

    def get_buffer_thres(self):
        thres = np.random.uniform(*self.threshold)
        buf = np.random.randint(*self.buffer)
        return buf, thres

    def get_params_dependent_on_targets(self, params):
        _img = params['image']
        if len(_img.shape) == 3:
            img = _img.max(2)
        else:
            img = _img
        buffer, threshold = self.get_buffer_thres()
        y_max, x_max = img.shape
        img2 = img > img.mean()
        y_mean = img2.mean(1)
        x_mean = img2.mean(0)
        x_mean[:5] = 0
        x_mean[-5:] = 0
        y_mean[:5] = 0
        y_mean[-5:] = 0
        y_mean = (y_mean - y_mean.min() + 1e-4) / (y_mean.max() - y_mean.min() + 1e-4)
        x_mean = (x_mean - x_mean.min() + 1e-4) / (x_mean.max() - x_mean.min() + 1e-4)
        y_slice = np.where(y_mean > threshold)[0]
        x_slice = np.where(x_mean > threshold)[0]
        if len(x_slice) == 0:
            x_start, x_end = 0, x_max
        else:
            x_start, x_end = max(x_slice.min() - buffer, 0), min(x_slice.max() + buffer, x_max)
        if len(y_slice) == 0:
            y_start, y_end = 0, y_max
        else:
            y_start, y_end = max(y_slice.min() - buffer, 0), min(y_slice.max() + buffer, y_max)
        return {"x_min": x_start, "x_max": x_end, "y_min": y_start, "y_max": y_end}

    def apply(self, img, x_min=0, y_min=0, x_max=0, y_max=0, **params):
        return img[y_min:y_max, x_min:x_max]

    def apply_to_mask(self, mask, x_min=0, y_min=0, x_max=0, y_max=0, **params):
        return mask[y_min:y_max, x_min:x_max, :]

    def apply_to_bbox(self, bbox, x_min=0, y_min=0, x_max=0, y_max=0, **params): # TODO
        return bbox
    
    @property
    def targets_as_params(self):
        return ["image"]
    
    def get_transform_init_args_names(self):
        return ('threshold', 'buffer')


class CropBBox(DualTransform):

    def __init__(self, buffer=30, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.buffer = buffer

    def apply(
        self, img: np.ndarray, x_min: int = 0, x_max: int = 0, y_min: int = 0, y_max: int = 0, **params
    ) -> np.ndarray:
        return img[y_min:y_max, x_min:x_max]

    def get_params_dependent_on_targets(self, params):
        img_y, img_x = params['image'].shape[:2]
        bbox = params['bboxes'][0]
        x_min = int(img_x*bbox[0]) - self.buffer
        x_max = int(img_x*bbox[2]) + self.buffer
        y_min = int(img_y*bbox[1]) - self.buffer
        y_max = int(img_y*bbox[3]) + self.buffer
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    
    def apply_to_bbox(self, bbox, **params):
        return bbox
    
    @property
    def targets_as_params(self):
        return ["image", "bboxes"]
    
    def get_transform_init_args_names(self):
        return {'buffer': self.buffer}


class RandomCropBBox(DualTransform):

    def __init__(self, buffer=(0, 120), always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.buffer = buffer

    def get_params_dependent_on_targets(self, params):
        img_y, img_x = params['image'].shape[:2]
        bbox = params['bboxes'][0]
        buf = np.random.randint(*self.buffer)
        x_min = int(img_x*bbox[0]) - buf
        x_max = int(img_x*bbox[2]) + buf
        y_min = int(img_y*bbox[1]) - buf
        y_max = int(img_y*bbox[3]) + buf
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def apply(
        self, img: np.ndarray, x_min: int = 0, x_max: int = 0, y_min: int = 0, y_max: int = 0, **params
    ) -> np.ndarray:
        return img[y_min:y_max, x_min:x_max]
    
    def apply_to_bbox(self, bbox, **params):
        return bbox
    
    @property
    def targets_as_params(self):
        return ["image", "bboxes"]
    
    def get_transform_init_args_names(self):
        return {'buffer': self.buffer}


class RandomCropBBox2(DualTransform):

    def __init__(self, buffer=(0, 120), always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.buffer = buffer

    def get_params_dependent_on_targets(self, params):
        img_y, img_x = params['image'].shape[:2]
        bbox = params['bboxes'][0]
        buf1 = np.random.randint(*self.buffer)
        buf2 = np.random.randint(*self.buffer)
        x_min = int(img_x*bbox[0]) - buf1
        x_max = int(img_x*bbox[2]) + buf1
        y_min = int(img_y*bbox[1]) - buf2
        y_max = int(img_y*bbox[3]) + buf2
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def apply(
        self, img: np.ndarray, x_min: int = 0, x_max: int = 0, y_min: int = 0, y_max: int = 0, **params
    ) -> np.ndarray:
        return img[y_min:y_max, x_min:x_max]
    
    def apply_to_bbox(self, bbox, **params):
        return bbox
    
    @property
    def targets_as_params(self):
        return ["image", "bboxes"]
    
    def get_transform_init_args_names(self):
        return {'buffer': self.buffer}


class AutoFlip(DualTransform):

    def __init__(self, sample_width=100, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.sample_width = sample_width
    
    def get_params_dependent_on_targets(self, params):
        img = params['image']
        if img[:, :self.sample_width].sum() <= img[:, -self.sample_width:].sum():
            flip = True
        else:
            flip = False
        return {"flip": flip}

    def apply(self, img, flip=False, **params):
        if flip:
            img = img[:, ::-1]
        return img

    def apply_to_mask(self, mask, flip=False, **params):
        if flip:
            mask = mask[:, ::-1, :]
        return mask

    def apply_to_bbox(self, bbox, flip=False, **params):
        return bbox
    
    @property
    def targets_as_params(self):
        return ["image"]
    
    def get_transform_init_args_names(self):
        return ('sample_width', 'sample_width')
    