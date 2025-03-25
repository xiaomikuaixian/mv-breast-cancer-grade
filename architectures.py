import torch.nn as nn
import timm
from train_utils.torch.modules import AdaptiveConcatPool2d, AdaptiveGeM

class MultiViewModel(nn.Module):
    def __init__(self,
                 classification_model='ConvNeXt',  # 分类器的基础模型，默认ConvNeXt
                 classification_params={},  # 模型额外参数
                 in_chans=1,  # 输入图像通道数
                 num_classes=1,  # 输出类别数量
                 num_view=2,  # 每个样本的视图数量
                 custom_classifier='concat',  # 同时考虑局部与全局特征
                 pool_view=False,  # 是否在视图维度上池化
                 dropout=0,  # dropout概率
                 hidden_dim=1024,  # 隐藏层维度
                 spatial_pool=True,  # 是否在空间维度上池化多个视图
                 pretrained=False,  # 是否加载预训练权重
                 freeze_layers=True,  # 是否冻结部分层
                 freeze_until=4):  # 冻结前几个卷积块

        super().__init__()

        # 初始化特征提取器
        self.encoder = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **classification_params
        )

        # 冻结特征提取器的指定层（适用于ConvNeXt等结构）
        if freeze_layers and hasattr(self.encoder, 'stages'):
            for i, stage in enumerate(self.encoder.stages):
                if i < freeze_until:
                    for param in stage.parameters():
                        param.requires_grad = False

        # 获取特征维度，并移除原始分类器
        feature_dim = self.encoder.get_classifier().in_features
        self.encoder.reset_classifier(0, '')

        self.attention = nn.Identity()  # 注意力模块，目前为空,下一版本迭代

        # 自定义全局池化方式
        if custom_classifier == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            self.global_pool = AdaptiveConcatPool2d()# 同时考虑局部与全局特征:同时进行全局平均池化和全局最大池化后再拼接，维度翻倍
            feature_dim *= 2
        elif custom_classifier == 'gem':
            self.global_pool = AdaptiveGeM(p=3, eps=1e-4)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 默认均值池化

        # 是否启用空间池化（将多个视图特征拼接在空间维度）
        self.spatial_pool = spatial_pool

        # 是否对视图维度进行池化
        self.pool_view = pool_view

        # 如果启用空间或视图池化，视图数量减半（特定应用场景下使用）
        if self.spatial_pool or self.pool_view:
            num_view = num_view // 2

        # 分类头网络结构定义
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),  # 展平特征
            nn.Linear(feature_dim*num_view, hidden_dim),  # 第一层全连接层
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  # Dropout正则化
            nn.Linear(hidden_dim, hidden_dim//2),  # 第二层全连接层
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  # Dropout正则化
            nn.Linear(hidden_dim//2, num_classes)  # 输出层
        )

    def forward(self, x):  # 输入维度：(批量大小 x 视图数 x 通道 x 宽 x 高)
        bs, n_view, ch, w, h = x.shape
        # 合并批量和视图维度进行特征提取
        x = x.view(bs*n_view, ch, w, h)
        y = self.encoder(x) # 提取特征

        # 如果启用空间池化，重塑特征张量拼接视图
        if self.spatial_pool:
            _, ch2, w2, h2 = y.shape
            y = y.view(bs, n_view, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
                .contiguous().view(bs, ch2, n_view*w2, h2)
        
        # 应用全局池化
        y = self.global_pool(y)

        # 如果没有空间池化，重塑特征为 (批量大小 x 视图数 x 特征)
        if not self.spatial_pool:
            y = y.view(bs, n_view, -1)
            # 如果启用视图池化，对视图维度求均值
            if self.pool_view:
                y = y.mean(1)

        # 通过分类头输出结果
        y = self.head(y)
        return y
