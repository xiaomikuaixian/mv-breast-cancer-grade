import torch.nn as nn
import timm
from train_utils.torch.modules import AdaptiveConcatPool2d, AdaptiveGeM

class MultiViewModel(nn.Module):
    def __init__(self,
                 classification_model='resnet18',
                 classification_params={},
                 in_chans=1,
                 num_classes=1,
                 num_view=2,
                 custom_classifier='none',
                 custom_attention='none', 
                 pool_view=False,
                 dropout=0,
                 hidden_dim=1024,
                 spatial_pool=False,
                 pretrained=False,
                 freeze_layers=True,
                 freeze_until=4):

        super().__init__()

        self.encoder = timm.create_model(
            classification_model,
            pretrained=False,
            in_chans=in_chans,
            num_classes=num_classes,
            **classification_params
        )

        if freeze_layers and hasattr(self.encoder, 'stages'):
            for i, stage in enumerate(self.encoder.stages):
                if i < freeze_until:
                    for param in stage.parameters():
                        param.requires_grad = False

        feature_dim = self.encoder.get_classifier().in_features
        self.encoder.reset_classifier(0, '')

        self.attention = nn.Identity()

        if custom_classifier == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            self.global_pool = AdaptiveConcatPool2d()
            feature_dim = feature_dim * 2
        elif custom_classifier == 'gem':
            self.global_pool = AdaptiveGeM(p=3, eps=1e-4)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.is_tranformer = False
        self.spatial_pool = spatial_pool
        self.pool_view = pool_view
        if self.spatial_pool or self.pool_view:
            num_view = num_view // 2

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feature_dim*num_view, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim//2), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim//2, num_classes))

    def forward(self, x): # (N x n_views x Ch x W x H)
        bs, n_view, ch, w, h = x.shape
        x = x.view(bs*n_view, ch, w, h)
        y = self.attention(self.encoder(x)) # (2n_views x Ch2 x W2 x H2)
        if self.is_tranformer:
            y = y.mean(dim=1).view(bs, n_view, -1).mean(dim=1)
        else:
            if self.spatial_pool:
                _, ch2, w2, h2 = y.shape
                y = y.view(bs, n_view, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
                    .contiguous().view(bs, ch2, n_view*w2, h2)
            y = self.global_pool(y) # (bs x Ch2 x 1 x 1)
        if not self.spatial_pool:
            y = y.view(bs, n_view, -1)
            if self.pool_view:
                y = y.mean(1)
        y = self.head(y)
        return y
