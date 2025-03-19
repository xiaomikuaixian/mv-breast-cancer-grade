import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.utils.data as D
import cv2

from general import *


class PatientLevelDataset(D.Dataset):
    """患者级别的乳腺癌图像数据集。
    
    该数据集按照患者ID和侧位(laterality)进行组织，支持从不同视图采样图像。
    
    Args:
        df: 包含患者和图像信息的DataFrame
        image_dir: 图像文件所在目录
        target_cols: 目标列名列表，默认为['grade_2_categ']
        aux_target_cols: 辅助目标列名列表
        metadata_cols: 元数据列名列表
        sep: 文件路径分隔符
        bbox_path: 包含边界框信息的文件路径
        preprocess: 预处理函数
        transforms: 数据增强转换函数
        flip_lr: 是否左右翻转
        sample_num: 每个视图类别采样的图像数量
        view_category: 视图类别分组列表
        replace: 采样时是否允许替换
        sample_criteria: 采样标准
        is_test: 是否为测试模式
        mixup_params: Mixup参数字典
        return_index: 是否返回索引
    """
    def __init__(
        self, 
        # 目标和元数据相关参数
        df, 
        image_dir, 
        target_cols=None,
        aux_target_cols=None, 
        metadata_cols=None, 
        sep='/', 

        # 路径和图像处理相关参数
        bbox_path=None, 
        preprocess=None, 
        transforms=None, 

        # 采样策略相关参数
        sample_num=1, 
        view_category=None, 
        replace=False, 
        sample_criteria='high_value', 
        is_test=False, 

        # 其他参数
        mixup_params=None, 
        return_index=False
    ):
        # 处理默认参数中的可变对象
        if target_cols is None:
            target_cols = ['grade_2_categ']
        if aux_target_cols is None:
            aux_target_cols = []
        if metadata_cols is None:
            metadata_cols = []
        if view_category is None:
            view_category = [['MLO'], ['CC']]
        
        self.df = df
        if 'oversample_id' in df.columns:
            self.df_dict = {pid: pdf for pid, pdf in df.groupby(['oversample_id', 'patient_id', 'laterality'])}
        else:
            self.df_dict = {pid: pdf for pid, pdf in df.groupby(['patient_id', 'laterality'])}
        self.pids = list(self.df_dict.keys())
        self.image_dir = image_dir
        self.target_cols = target_cols
        self.aux_target_cols = aux_target_cols
        self.metadata_cols = metadata_cols
        if bbox_path is None:
            self.bbox = None
        else:
            self.bbox = pd.read_csv(bbox_path).set_index('name').to_dict(orient='index')
        self.preprocess = preprocess
        self.transforms = transforms
        self.is_test = is_test
        self.sample_num = sample_num
        self.view_category = view_category
        self.replace = replace
        self.sample_criteria = sample_criteria

        # 针对测试集的数据采样策略
        assert sample_criteria in ['high_value', 'low_value_for_implant', 'latest', 'valid_area']
        if mixup_params:
            assert 'alpha' in mixup_params.keys()
            self.mu = True
            self.mu_a = mixup_params['alpha']
        else:
            self.mu = False
        self.rt_idx = return_index
        self.sep = sep

    def update_df(self, new_df):
        """更新数据集的DataFrame。
        
        Args:
            new_df: 要添加的新DataFrame
        """
        self.df = pd.concat([self.df, new_df])
        if 'oversample_id' in self.df.columns:
            self.df_dict = {pid: pdf for pid, pdf in self.df.groupby(['oversample_id', 'patient_id', 'laterality'])}
        else:
            self.df_dict = {pid: pdf for pid, pdf in self.df.groupby(['patient_id', 'laterality'])}
        self.pids = list(self.df_dict.keys())

    def __len__(self):
        """返回数据集中患者的数量。"""
        return len(self.df_dict)  # 患者数量

    def _process_img(self, img, bbox=None):
        """处理图像，应用预处理和转换。
        
        Args:
            img: 输入图像
            bbox: 边界框信息
            
        Returns:
            处理后的图像
        """

        if self.preprocess:
            if bbox is None:
                # 输出图像的维度
                img = self.preprocess(image=img)['image']
            else:
                img_h, img_w = img.shape
                bbox[2] = min(bbox[2], img_h)
                bbox[3] = min(bbox[3], img_w)
                img = self.preprocess(image=img, bboxes=[bbox])['image']

        if self.transforms:
            if len(img.shape) == 4:  # Tile x W x H x Ch
                output = []
                for tile in img:
                    output.append(self.transforms(image=tile)['image'])  # -> torch
                output = torch.stack(output)
                img = output
            else:  # W x H x Ch
                img = self.transforms(image=img)['image']
        
        return img

    def _load_image(self, path, bbox=None):
        """加载并处理图像。
        
        Args:
            path: 图像文件路径
            bbox: 边界框信息
            
        Returns:
            处理后的图像
        """
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._process_img(img, bbox=bbox)
        return img

    def _get_file_path(self, patient_id, image_id):
        """获取图像文件的完整路径。
        
        Args:
            patient_id: 患者ID
            image_id: 图像ID
            
        Returns:
            图像文件的完整路径
        """
        return self.image_dir/f'{patient_id}{self.sep}{image_id}.png'

    def _get_bbox(self, patient_id, image_id):
        """获取图像的边界框信息。
        
        Args:
            patient_id: 患者ID
            image_id: 图像ID
            
        Returns:
            边界框信息，如果不存在则返回None或默认值
        """
        if self.bbox is not None:
            key = f'{patient_id}/{image_id}.png'
            if key in self.bbox.keys():
                bbox = self.bbox[key]
                bbox = [bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax'], 'YOLO']
            else:
                bbox = [0, 0, 100000, 100000, 'YOLO']  # 默认边界框
        else:
            bbox = None
        return bbox

    def _load_best_image(self, df):
        """根据采样标准加载最佳图像(用于测试集)。
        
        Args:
            df: 包含图像信息的DataFrame
            
        Returns:
            处理后的最佳图像列表和对应的图像ID列表
        """
        if self.sample_criteria == 'latest':
            try:
                latest_idx = np.argmax(df['content_date'])
                df2 = df.iloc[latest_idx:latest_idx+1]
            except:
                pass
        else:
            df2 = df
        scores = []
        images = []
        bboxes = []
        iids = []
        if 'implant' in df2.columns:
            is_implant = df2['implant'].values[0]
        else:
            is_implant = 0
        for pid, iid in df2[['patient_id', 'image_id']].values:
            img_path = self._get_file_path(pid, iid)
            bbox = self._get_bbox(pid, iid)
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 根据不同标准计算图像得分
            if self.sample_criteria == 'high_value':
                score = img.mean()
            elif self.sample_criteria == 'low_value_for_implant':
                score = img.mean()
                if is_implant:
                    score = 1 / (score + 1e-4)
            elif self.sample_criteria == 'valid_area':
                score = ((16 < img) & (img < 160)).mean()
                if is_implant:
                    score = 1 / (score + 1e-4)
                    
            bboxes.append(bbox)
            scores.append(score)
            images.append(img)
            iids.append(iid)
            
        score_idx = np.argsort(scores)[::-1]
        output_imgs = [self._process_img(images[idx], bboxes[idx]) for idx in score_idx[:self.sample_num]]
        return output_imgs, [iids[idx] for idx in score_idx[:self.sample_num]]

    def _load_data(self, idx):
        """加载指定索引的数据。
        
        Args:
            idx: 数据索引
            
        Returns:
            图像张量和标签张量
        """
        pid = self.pids[idx]
        pdf = self.df_dict[pid]
        img = []
        img_ids = []  # 记录已采样的图像ID
        print(idx)
        print(self.view_category)
        print(pdf)
        
        # 从每个视图类别中采样图像
        for iv, view_cat in enumerate(self.view_category):
            view0 = pdf.loc[pdf['view'].isin(view_cat) & ~pdf['image_id'].isin(img_ids)]
            print(view0)
            if not self.replace and len(view0) == 0:
                view0 = pdf.loc[pdf['view'].isin(view_cat)]
            if len(view0) == 0:
                img0 = []
            else:
                if self.is_test:
                    img0, iid = self._load_best_image(view0)
                    if not self.replace:
                        img_ids.extend(iid)
                else:
                    view0 = view0.sample(min(self.sample_num, len(view0)))
                    img0 = []
                    for pid, iid in view0[['patient_id', 'image_id']].values:
                        print(pid, iid)
                        img_path = self._get_file_path(pid, iid)
                        bbox = self._get_bbox(pid, iid)
                        img0.append(self._load_image(img_path, bbox))
                        if not self.replace:
                            img_ids.append(iid)
            img.extend(img0)
        print(img)
        img = torch.stack(img, dim=0)
        expected_dim = self.sample_num * len(self.view_category)
        if img.shape[0] < expected_dim:
            # 如果图像数量不足，用零填充
            img = torch.concat(
                [img, torch.zeros(
                    (expected_dim-img.shape[0], *img.shape[1:]), dtype=torch.float32)], dim=0)

        label = torch.from_numpy(pdf[self.target_cols+self.aux_target_cols].values[0].astype(np.float16))
        
        return img, label

    def __getitem__(self, idx):
        """获取数据集中指定索引的样本。
        
        Args:
            idx: 样本索引
            
        Returns:
            样本图像、标签以及可能的索引
        """
        img, label = self._load_data(idx)

        if self.mu:
            idx2 = np.random.randint(0, len(self.images))
            lam = np.random.beta(self.mu_a, self.mu_a)
            img2, label2 = self._load_data(idx2)
            img = lam * img + (1 - lam) * img2
            label = lam * label + (1 - lam) * label2

        if self.rt_idx:
            return img, label, idx
        else:
            return img, label

    def get_labels(self):
        """获取数据集中所有样本的标签。
        
        Returns:
            标签数组
        """
        labels = []
        for idx in range(len(self.df_dict)):
            pid = self.pids[idx]
            pdf = self.df_dict[pid]
            labels.append(pdf[self.target_cols].values[0].reshape(1, 1).astype(np.float16))
        return np.concatenate(labels, axis=0)