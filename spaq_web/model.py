import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights


class EXIFFeatureExtractor(nn.Module):
    """EXIF特征提取模块"""
    
    def __init__(self, input_dim=8, hidden_dim=64):
        """
        参数:
            input_dim: EXIF特征输入维度
            hidden_dim: 隐藏层维度
        """
        super(EXIFFeatureExtractor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.network(x)


class SPAQRegressor(nn.Module):
    """
    基于迁移学习的图像质量评分模型（支持EXIF特征融合）
    使用预训练的ResNet50，融合EXIF特征，输出1个分数
    """
    
    def __init__(self, pretrained=True, freeze_backbone=False, use_exif=False):
        """
        参数:
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结卷积层
            use_exif: 是否使用EXIF特征融合
        """
        super(SPAQRegressor, self).__init__()
        
        self.use_exif = use_exif
        
        # 加载预训练ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 获取全连接层的输入特征数
        num_features = self.backbone.fc.in_features  # 2048
        
        # 冻结卷积层参数
        if freeze_backbone:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer2.parameters():
                param.requires_grad = False
            for param in self.backbone.layer3.parameters():
                param.requires_grad = False
            for param in self.backbone.layer4.parameters():
                param.requires_grad = False
        
        # 移除原有的全连接层，只保留特征提取部分
        self.backbone.fc = nn.Identity()
        
        if use_exif:
            # EXIF特征提取器
            self.exif_extractor = EXIFFeatureExtractor(input_dim=8, hidden_dim=64)
            
            # 特征融合后的回归头
            # 输入：2048 (图像) + 64 (EXIF) = 2112
            fusion_input_dim = num_features + 64
            
            self.regression_head = nn.Sequential(
                nn.Linear(fusion_input_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
        else:
            # 原有回归头（无EXIF）
            self.regression_head = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
    
    def forward(self, image, exif_features=None):
        """
        前向传播
        
        参数:
            image: 图像张量 (B, 3, 224, 224)
            exif_features: EXIF特征张量 (B, 8)，仅当use_exif=True时需要
        """
        # 图像特征提取
        image_features = self.backbone(image)
        
        if self.use_exif:
            assert exif_features is not None, "use_exif=True时必须提供exif_features"
            
            # EXIF特征提取
            exif_embedded = self.exif_extractor(exif_features)
            
            # 特征融合（拼接）
            fused_features = torch.cat([image_features, exif_embedded], dim=1)
        else:
            fused_features = image_features
        
        # 回归预测
        output = self.regression_head(fused_features)
        
        return output


def create_model(pretrained=True, freeze_backbone=False, use_exif=False):
    """
    创建模型的便捷函数
    
    参数:
        pretrained: 是否使用预训练权重
        freeze_backbone: 是否冻结卷积层
        use_exif: 是否使用EXIF特征融合
    """
    model = SPAQRegressor(
        pretrained=pretrained, 
        freeze_backbone=freeze_backbone,
        use_exif=use_exif
    )
    return model