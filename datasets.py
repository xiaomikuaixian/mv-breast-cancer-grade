import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.utils.data as D
import cv2
from general import *
import matplotlib.pyplot as plt

class PatientLevelDataset(D.Dataset):
    """患者级别的乳腺癌图像数据集。

    该数据集按患者ID和侧位(laterality)组织，支持从不同视图采样图像。
    每个样本包含来自同一患者同一侧乳房的一个或多个视图的图像。

    Args:
        df (pd.DataFrame): 包含患者和图像信息的DataFrame。
                           必须包含 'patient_id', 'laterality', 'image_id', 'view' 列。
        image_dir (str or Path): 图像文件所在的根目录。
        target_cols (list[str], optional): 目标列名列表。默认为 ['grade_2_categ']。
        aux_target_cols (list[str], optional): 辅助目标列名列表。默认为空列表。
        sep (str, optional): 构建图像路径时 patient_id 和 image_id 之间的分隔符。默认为 '/'。
        bbox_path (str or Path, optional): 包含边界框信息的CSV文件路径。默认为 None。
        preprocess (callable, optional): 应用于原始图像的预处理函数（例如，调整大小、归一化）。
                                         应接受关键字参数 'image' 和可选的 'bboxes'。
                                         默认为 None。
        transforms (callable, optional): 应用于预处理后图像的数据增强函数。
                                         应接受关键字参数 'image'。默认为 None。
        sample_num (int, optional): 每个视图类别要采样的图像数量。默认为 1。
        view_category (list[list[str]], optional): 定义视图分组的列表。
                                                   例如 [['MLO'], ['CC']]。默认为 [['MLO'], ['CC']]。
        replace (bool, optional): 在同一视图类别内采样时是否允许重复选择同一图像。默认为 False。
        sample_criteria (str, optional): 测试时选择图像的标准 ('high_value', 'valid_area')。
                                         仅在 is_test=True 时生效。默认为 'high_value'。
        is_test (bool, optional): 是否为测试模式。测试模式下通常不进行随机采样和增强，
                                  并可能使用 sample_criteria 选择图像。默认为 False。
        return_index (bool, optional): __getitem__ 是否额外返回样本索引。默认为 False。
    """
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir,
        target_cols = None,
        aux_target_cols = None,
        sep: str = '/',
        bbox_path = None,
        preprocess = None,
        transforms = None,
        sample_num: int = 1,
        view_category: list[list[str]] = None,
        replace: bool = False,
        sample_criteria: str = 'high_value',
        is_test: bool = False,
        return_index: bool = False
    ):
        # --- 1. 初始化参数处理 ---
        # 处理默认的可变参数
        self.target_cols = ['grade_2_categ'] if target_cols is None else target_cols
        self.aux_target_cols = [] if aux_target_cols is None else aux_target_cols
        self.view_category = [['MLO'], ['CC']] if view_category is None else view_category

        self.df = df
        self.image_dir = Path(image_dir) # 确保是 Path 对象
        self.sep = sep
        self.preprocess = preprocess
        self.transforms = transforms
        self.sample_num = sample_num
        self.replace = replace
        self.is_test = is_test
        self.return_index = return_index

        # 测试模式下的采样标准
        valid_criteria = ['high_value', 'valid_area']
        if self.is_test:
            if sample_criteria not in valid_criteria:
                raise ValueError(f"sample_criteria '{sample_criteria}'无效，测试模式下必须是 {valid_criteria} 之一")
            self.sample_criteria = sample_criteria

        # 训练模式下忽略 sample_criteria，进行随机采样
        elif sample_criteria != 'high_value': # 如果不是测试模式，但设置了非默认值，给个提醒
             print(f"警告: is_test=False，sample_criteria ('{sample_criteria}') 将被忽略，使用随机采样。")
             self.sample_criteria = 'high_value' # 设回默认，虽然不用

        # --- 2. 组织数据 ---
        # 根据 'patient_id' 和 'laterality'对 DataFrame 进行分组
        group_cols = ['patient_id', 'laterality']
        # self.df_dict 的键是分组列构成的元组，值是对应的子 DataFrame
        self.df_dict = {pid: pdf for pid, pdf in df.groupby(group_cols)}
        # self.pids 是所有唯一的(patient_id, laterality) 元组列表
        self.pids = list(self.df_dict.keys())

        # --- 3. 加载边界框信息 (如果提供) ---
        if bbox_path:
            try:
                bbox_df = pd.read_csv(bbox_path)
                # 假设CSV有 'name', 'ymin', 'xmin', 'ymax', 'xmax' 列
                # 'name' 通常是 'patient_id/image_id.png' 的形式
                self.bbox_dict = bbox_df.set_index('name').to_dict(orient='index')
            except FileNotFoundError:
                print(f"警告: 无法找到边界框文件 {bbox_path}。将不使用边界框。")
                self.bbox_dict = None
            except KeyError as e:
                print(f"警告: 边界框文件 {bbox_path} 缺少列 {e}。将不使用边界框。")
                self.bbox_dict = None
        else:
            self.bbox_dict = None # 没有提供路径，则不使用边界框

    def __len__(self) -> int:
        """返回数据集中唯一的 患者-侧位 组合的数量。"""
        return len(self.pids)

    def _get_file_path(self, patient_id, image_id) -> Path:
        """构建并返回图像文件的完整路径。"""
        # 使用 Path 对象拼接路径更安全可靠
        return self.image_dir / f'{patient_id}{self.sep}{image_id}.png'

    def _get_bbox(self, patient_id, image_id) -> list | None:
        """获取指定图像的边界框信息。"""
        if self.bbox_dict is None:
            return None # 没有加载边界框数据

        key = f'{patient_id}/{image_id}.png'
        if key in self.bbox_dict:
            bbox_data = self.bbox_dict[key]
            # 返回 [ymin, xmin, ymax, xmax, label] 格式，label 可以是任意字符串，常用于某些库
            return [bbox_data['ymin'], bbox_data['xmin'], bbox_data['ymax'], bbox_data['xmax'], 'ROI']
        else:
            return None # 或者可以返回一个覆盖整个图像的默认框，但这取决于 preprocess 如何处理 None 或默认框

    def _process_img(self, img: np.ndarray, bbox: list | None = None) -> torch.Tensor:
        """应用预处理和数据增强变换。"""
        processed_img = img

        # 1. 应用预处理 (通常是大小调整、归一化等)
        if self.preprocess:
            if bbox:
                # 确保 bbox 坐标在图像范围内 (以防标注错误或图像被裁剪)
                img_h, img_w = processed_img.shape[:2]
                ymin, xmin, ymax, xmax, label = bbox
                ymin = max(0, ymin)
                xmin = max(0, xmin)
                ymax = min(img_h, ymax)
                xmax = min(img_w, xmax)
                # 检查坐标有效性
                if ymax <= ymin or xmax <= xmin:
                     print(f"警告: 图像处理中遇到无效边界框坐标 (原始: {bbox}, 裁剪后: {[ymin, xmin, ymax, xmax]})，将忽略此边界框。")
                     processed_img = self.preprocess(image=processed_img)['image'] # 不带 bbox 处理
                else:
                    valid_bbox = [ymin, xmin, ymax, xmax, label]
                    processed_img = self.preprocess(image=processed_img, bboxes=[valid_bbox])['image']
            else:
                 processed_img = self.preprocess(image=processed_img)['image']

        # 2. 应用数据增强 (通常是几何变换、颜色抖动等)

        # 仅在训练模式下应用增强
        if self.transforms and not self.is_test:
            
            # 假设 preprocess 输出是 H x W x C 或 H x W
            # 确保输入是 NumPy 数组，因为 Albumentations 通常期望这个
            if isinstance(processed_img, torch.Tensor):
                 processed_img = processed_img.numpy() # 需要根据具体情况调整，例如 .permute().numpy()

            # preprocess 输出 HxWxC 或 HxW
            if len(processed_img.shape) == 3 or len(processed_img.shape) == 2: # HxWxC 或 HxW
                processed_img = self.transforms(image=processed_img)['image']

        # 3. 确保输出是 Tensor
        if not isinstance(processed_img, torch.Tensor):
            # 典型的转换，如果图像是 HxWxC (NumPy) -> CxHxW (Tensor)
            if len(processed_img.shape) == 3: # HxWxC
                 processed_img = torch.from_numpy(processed_img.transpose((2, 0, 1))).float()
            elif len(processed_img.shape) == 2: # HxW (grayscale)
                 processed_img = torch.from_numpy(processed_img).unsqueeze(0).float() # -> 1xHxW

        # 简单的类型转换，确保是 float32
        if processed_img.dtype != torch.float32:
             processed_img = processed_img.float()

        return processed_img

    def _load_image(self, patient_id, image_id) -> torch.Tensor:
        """加载、处理单张图像并返回 Tensor。"""
        img_path = self._get_file_path(patient_id, image_id)
        bbox = self._get_bbox(patient_id, image_id)

        try:
            # 以灰度模式读取图像
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"无法加载图像: {img_path}")
        except Exception as e:
            print(f"错误: 加载或读取图像 {img_path} 时出错: {e}")
            # 返回一个占位符或引发错误，这里返回一个小的黑色图像
            return torch.zeros((1, 64, 64), dtype=torch.float32) # 示例：返回 1x64x64 黑色图像

        # 处理图像（预处理 + 增强）
        img_tensor = self._process_img(img, bbox=bbox)
        return img_tensor

    def _load_best_image(self, df_view: pd.DataFrame) -> tuple[list[torch.Tensor], list]:
        """根据采样标准为测试集加载最佳图像。"""
        candidate_df = df_view.copy() # 操作副本

        # 1. 特殊情况：按最新日期筛选 (如果标准是 'latest')
        if self.sample_criteria == 'latest' and 'content_date' in candidate_df.columns:
            try:
                # 假设 content_date 可以被解析为日期或已经是可比较的格式
                latest_idx = pd.to_datetime(candidate_df['content_date'], errors='coerce').idxmax()
                if pd.notna(latest_idx):
                    candidate_df = candidate_df.loc[[latest_idx]]
                else:
                    print(f"警告: 无法确定 patient {candidate_df['patient_id'].iloc[0]} 的最新图像，将使用所有候选图像。")
            except Exception as e:
                print(f"警告: 处理 'content_date' 时出错 ({e})，将使用所有候选图像。")

        # 2. 为每个候选图像计算分数
        scores = []
        raw_images = [] # 存储原始加载的图像 (numpy)
        bboxes = []     # 存储对应的边界框
        image_ids = []  # 存储图像 ID

        # 检查植入物状态 (假设同一患者侧位的植入物状态一致)
        has_implant_col = 'implant' in candidate_df.columns
        is_implant = candidate_df['implant'].iloc[0] == 1 if has_implant_col else False

        for _, row in candidate_df.iterrows():
            pid = row['patient_id']
            iid = row['image_id']
            img_path = self._get_file_path(pid, iid)
            bbox = self._get_bbox(pid, iid)

            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"警告: 测试集加载最佳图像时，无法加载 {img_path}，跳过此图像。")
                    continue # 跳过无法加载的图像

                # 计算分数
                if self.sample_criteria == 'high_value':
                    score = img.mean()
                elif self.sample_criteria == 'low_value_for_implant':
                    # 对于有植入物的，偏好平均亮度低的（分数高）；无植入物的，偏好亮度高的。
                    mean_val = img.mean()
                    score = 1.0 / (mean_val + 1e-6) if is_implant else mean_val
                elif self.sample_criteria == 'valid_area':
                    # 计算有效区域占比（例如，代表组织的像素范围）
                    valid_pixels = ((16 < img) & (img < 240)).mean() # 示例范围，可调整
                    # 对于有植入物的，偏好有效区域占比低的（分数高）；无植入物的，偏好占比高的。
                    score = 1.0 / (valid_pixels + 1e-6) if is_implant else valid_pixels
                elif self.sample_criteria == 'latest':
                     score = 1 # 如果按日期筛选后只剩一个，分数不重要；否则需要 fallback 或不同逻辑
                     # 如果 fallback 到多张图像，可能需要结合其他标准，这里简单处理
                else: # latest 且无法筛选时，或 fallback
                     score = img.mean() # 默认回退到 high_value

                raw_images.append(img)
                scores.append(score)
                bboxes.append(bbox)
                image_ids.append(iid)

            except Exception as e:
                print(f"错误: 在 _load_best_image 中处理图像 {img_path} 时出错: {e}")
                continue

        if not raw_images: # 如果所有候选图像都加载失败
             return [], []

        # 3. 根据分数排序并选择 top N
        # argsort 返回的是索引，[::-1] 实现降序
        sorted_indices = np.argsort(scores)[::-1]
        num_to_select = min(self.sample_num, len(raw_images))
        selected_indices = sorted_indices[:num_to_select]

        # 4. 处理选定的图像 (预处理，但不应用训练时增强)
        output_imgs = []
        selected_iids = []
        original_test_state = self.is_test
        self.is_test = True # 强制设为 True，确保 _process_img 不应用 transforms
        try:
            for idx in selected_indices:
                processed_img = self._process_img(raw_images[idx], bboxes[idx])
                output_imgs.append(processed_img)
                selected_iids.append(image_ids[idx])
        finally:
             self.is_test = original_test_state # 恢复原始状态

        return output_imgs, selected_iids

    def _load_data(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """加载指定索引 (患者-侧位对) 的数据。"""

        # 1. 获取该患者-侧位对应的 DataFrame
        pid_key = self.pids[idx]
        patient_df = self.df_dict[pid_key]

        # 2. 按视图类别采样图像
        selected_images = [] # 存储最终选中的图像 Tensor
        sampled_image_ids = set() # 用于 'replace=False' 时的去重

        # 预期每个样本的总图像数
        expected_num_images = self.sample_num * len(self.view_category)

        for view_cat in self.view_category:
            # 筛选出属于当前视图类别的图像
            view_mask = patient_df['view'].isin(view_cat)
            candidate_df = patient_df[view_mask].copy() # 使用副本

            # 如果不允许重复，并且之前已采样过图像，则排除这些图像
            if not self.replace and sampled_image_ids:
                candidate_df = candidate_df[~candidate_df['image_id'].isin(sampled_image_ids)]

            # 如果筛选后没有图像（可能因为视图缺失，或不允许重复时图像已用完）
            if candidate_df.empty:
                # 如果不允许重复导致为空，尝试允许重复选择（从原始视图中选）
                if not self.replace and patient_df[view_mask].shape[0] > 0:
                    print(f"信息: 患者 {pid_key[:2]} 视图 {view_cat} 无剩余可选图像 (replace=False)，将从该视图所有图像中重新选择。")
                    candidate_df = patient_df[view_mask].copy() # 退回原始候选集
                else:
                    # 确实没有该视图的图像，或即使允许重复也选完了
                    print(f"警告: 患者 {pid_key[:2]} 缺少视图 {view_cat} 或图像不足。")
                    # 不需要添加空图像，后续会用零填充
                    continue # 进行下一个视图类别的采样

            # 确定实际要采样的数量
            num_available = len(candidate_df)
            num_to_sample = min(self.sample_num, num_available)

            images_for_this_view = []
            sampled_ids_this_view = []

            if self.is_test:
                # 测试模式：使用 _load_best_image 选择图像
                images_for_this_view, sampled_ids_this_view = self._load_best_image(candidate_df)
                # _load_best_image 内部已处理了 sample_num
            else:
                # 训练模式：随机采样
                # replace 参数在这里控制 sample 是否有放回
                sampled_rows = candidate_df.sample(n=num_to_sample, replace=self.replace)
                for _, row in sampled_rows.iterrows():
                    img_tensor = self._load_image(row['patient_id'], row['image_id'])
                    images_for_this_view.append(img_tensor)
                    sampled_ids_this_view.append(row['image_id'])

            selected_images.extend(images_for_this_view)
            if not self.replace: # 如果不允许重复，记录本次采样的 ID
                sampled_image_ids.update(sampled_ids_this_view)


        # 3. 组合图像张量并进行填充
        if not selected_images:
            # 如果一个图像都没选到（极其罕见的情况，比如所有图像都加载失败）
            print(f"错误: 患者 {pid_key[:2]} 未能加载任何有效图像！将返回零张量。")

            # 获取一个示例图像的形状用于填充，或者使用预定义形状
            try:
                # 尝试加载同组的第一个图像获取形状
                 example_row = patient_df.iloc[0]
                 example_img = self._load_image(example_row['patient_id'], example_row['image_id'])
                 img_shape = example_img.shape
            except:
                 img_shape = (1, 256, 256) # 预定义的默认形状
            img_tensor = torch.zeros((expected_num_images, *img_shape), dtype=torch.float32)
        else:
            img_tensor = torch.stack(selected_images, dim=0)
            current_num_images = img_tensor.shape[0]

            # 如果图像数量不足预期，用零向量填充
            if current_num_images < expected_num_images:
                num_padding = expected_num_images - current_num_images
                padding_tensor = torch.zeros((num_padding, *img_tensor.shape[1:]), dtype=torch.float32)
                img_tensor = torch.cat([img_tensor, padding_tensor], dim=0)

            # 如果图像数量超过预期（理论上不应发生，除非 sample_num * len(view_category) 计算错误或逻辑问题）
            elif current_num_images > expected_num_images:
                 print(f"警告: 患者 {pid_key[:2]} 加载的图像数 ({current_num_images}) 超出预期 ({expected_num_images})，将截断。")
                 img_tensor = img_tensor[:expected_num_images]


        # 4. 提取标签
        # 同一患者-侧位的所有图像共享相同的标签
        label_values = patient_df[self.target_cols + self.aux_target_cols].iloc[0].values

        # 转换为 float16 以节省内存
        label_tensor = torch.from_numpy(label_values.astype(np.float16))

        return img_tensor, label_tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, int]:
        """获取数据集中指定索引的样本。"""

        # 加载图像和标签数据
        img_tensor, label_tensor = self._load_data(idx)

        # 根据需要返回索引
        if self.return_index:
            return img_tensor, label_tensor, idx
        else:
            return img_tensor, label_tensor

    def get_labels(self) -> np.ndarray:
        """获取数据集中所有样本的主目标标签。"""
        all_labels = []
        for pid_key in self.pids:
            # 获取该患者-侧位对应的第一个图像行的标签值
            label_values = self.df_dict[pid_key][self.target_cols].iloc[0].values
            all_labels.append(label_values)
        # 将标签列表转换为 NumPy 数组
        return np.array(all_labels, dtype=np.float16) # 使用 float16 与内部一致

def visualize_batch(dataloader, batch_idx=0, save_path=None, figsize_per_img=(10, 10), dpi=300):
    """可视化数据加载器中的一个批次图像。

    Args:
        dataloader: 包含图像批次的数据加载器
        batch_idx: 要可视化的批次索引，默认为0（第一个批次）
        save_path: 保存图像的路径，如果为None则不保存
        figsize_per_img: 每个子图的大小，默认为(10, 10)
        dpi: 图像分辨率，默认为300
    """
    # 获取指定批次
    for i, batch in enumerate(dataloader):
        if i != batch_idx:
            continue
            
        if len(batch) == 3:  # 如果batch包含索引
            images, labels, indices = batch
        else:
            images, labels = batch
            indices = list(range(len(images)))
            
        batch_size = images.shape[0]
        num_views = images.shape[1]  # 每个样本的视图数量
        
        # 创建子图网格
        fig, axes = plt.subplots(batch_size, num_views, 
                                figsize=(num_views*figsize_per_img[0], batch_size*figsize_per_img[1]), 
                                dpi=dpi)
        
        # 处理单样本或单视图的情况
        if batch_size == 1 and num_views == 1:
            axes = np.array([[axes]])
        elif batch_size == 1:
            axes = axes.reshape(1, -1)
        elif num_views == 1:
            axes = axes.reshape(-1, 1)
        
        # 遍历批次中的每个样本和每个视图
        for b in range(batch_size):
            for v in range(num_views):
                ax = axes[b, v]
                img = images[b, v]  # 获取当前样本的当前视图
                
                # 将张量转换为NumPy数组
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                
                # 处理不同的通道排列
                if img.shape[0] in [1, 3]:  # CxHxW格式
                    img = np.transpose(img, (1, 2, 0))
                
                # 如果是单通道图像，去掉通道维度

                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                
                # 计算像素值范围
                min_val = img.min()
                max_val = img.max()
                
                # 直接显示图像，不做额外处理
                ax.imshow(img, cmap='gray')
                ax.set_title(f"sample{indices[b]}, view{v}\nsize: {img.shape[0]}x{img.shape[1]}\nrange: [{min_val:.2f}, {max_val:.2f}]", fontsize=12)
                ax.axis('off')
        
        plt.tight_layout()
        
        # 保存高分辨率图像
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"图像已保存至: {save_path}")
            
        return fig  # 返回图形对象以便进一步处理
        
    print(f"未找到索引为 {batch_idx} 的批次")
    return None

if __name__ == '__main__':
    # 读取测试样例
    dummy_df = pd.read_csv('./input/BC_MG/test_debug.csv',header=0,sep=',')
    # 测试图像存放目录
    dummy_image_dir = Path('./input/BC_MG/image_resized_2048') # 使用 Path 对象

    # --- 定义预处理和增强 (可选) ---
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from transforms import AutoFlip, CropROI
    
    preprocess_fn = A.Compose([
            CropROI(buffer=80),
            AutoFlip(sample_width=100),
            A.Resize(1536, 768)  # 1536×768，捕获更多细节
        ], bbox_params=A.BboxParams(format='pascal_voc'))
    # preprocess_fn = None
    
    transforms_fn = A.Compose([
        # Add normalization first if not done already
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
    ])
    # transforms_fn = None


    print("\n--- 创建训练数据集 ---")
    train_dataset = PatientLevelDataset(
        df=dummy_df,
        image_dir=dummy_image_dir,
        target_cols=['grade_2_categ'],
        sample_num=1, # 每个视图类别取1张
        view_category=[['MLO'], ['CC']],
        replace=False,
        is_test=False,
        preprocess=preprocess_fn,
        transforms=transforms_fn,
        return_index=True
    )

    print(f"数据集大小 (患者-侧位对): {len(train_dataset)}")

    # 获取第一个样本
    if len(train_dataset) > 0:
        img_tensor, label_tensor, index = train_dataset[0]
        print(f"\n获取样本索引 {index}:")
        print(f"  图像张量形状: {img_tensor.shape}") # 应该是 (sample_num * num_views, C, H, W) 或 (sample_num * num_views, H, W)
        print(f"  图像张量类型: {img_tensor.dtype}")
        print(f"  标签张量: {label_tensor}")
        print(f"  标签张量类型: {label_tensor.dtype}")

    # 创建 DataLoader
    train_loader = D.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    print("\n--- 迭代 Train DataLoader ---")
    for i, batch in enumerate(train_loader):
        images, labels, indices = batch
        print(f"批次 {i}:")
        print(f"  图像批次形状: {images.shape}")
        print(f"  标签批次: {labels}")
        print(f"  索引: {indices}")
        if i >= 1: # 只显示前2个批次
             break
    
    # 添加图像可视化代码
    print("\n--- 可视化批次图像 ---")
    visualize_batch(train_loader, batch_idx=0, save_path='./transformed_batch_0.png')

    # print("\n--- 创建测试数据集 (使用 'high_value' 标准) ---")
    # test_dataset = PatientLevelDataset(
    #     df=dummy_df,

    #     target_cols=['grade_2_categ'],
    #     sample_num=1,
    #     view_category=[['MLO'], ['CC']],
    #     sample_criteria='high_value', 
    #     is_test=True,
    #     preprocess=preprocess_fn, # 测试时通常也需要预处理
    #     transforms=None, # 测试时通常不需要增强
    #     return_index=True
    # )

    # if len(test_dataset) > 0:
    #      img_tensor_test, label_tensor_test, index_test = test_dataset[0]
    #      print(f"\n获取测试样本 0:")
    #      print(f"  图像张量形状: {img_tensor_test.shape}")
    #      print(f"  标签张量: {label_tensor_test}")

    # # 创建 DataLoader
    # test_loader = D.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # print("\n--- 迭代 Test DataLoader ---")
    # for i, batch in enumerate(test_loader):
    #     images, labels, indices = batch
    #     print(f"批次 {i}:")
    #     print(f"  图像批次形状: {images.shape}")
    #     print(f"  标签批次: {labels}")
    #     print(f"  索引: {indices}")
    #     if i >= 1: # 只显示前2个批次
    #          break