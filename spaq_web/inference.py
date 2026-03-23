"""
SPAQ 模型推理引擎
负责模型的加载、图像预处理、EXIF 特征提取及单张图片的质量评分
"""

import os
import torch
from PIL import Image, ExifTags
from torchvision import transforms
import logging
import math
import exifread

# 导入你现有的模型构建函数
from model import create_model

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPAQPredictor:
    """SPAQ 图像质量预测器单例类"""
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SPAQPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path='checkpoints/best_model.pth'):
        if self._initialized:
            return
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_exif = True
        self.model_path = model_path
        
        # 你的 8 维 EXIF 特征归一化参数
        self.EXIF_MEAN = [3.97, 2.09, 0.023, 309.70, 1.50, 0.024, -0.47, -0.57]
        self.EXIF_STD = [1.27, 0.20, 0.034, 693.45, 3.04, 0.15, 0.52, 0.42]
        self.EXIF_DIM = 8
        
        self.transform = self._get_transform()
        self.model = self._load_model()
        self._initialized = True

    def _get_transform(self):
        """与验证集一致的图像变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self):
        """加载预训练模型权重"""
        logger.info(f"正在加载模型权重: {self.model_path} (Device: {self.device})")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"未找到模型权重文件: {self.model_path}")
            
        model = create_model(pretrained=False, freeze_backbone=False, use_exif=self.use_exif)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        logger.info("模型加载完成。")
        return model

    def _extract_and_normalize_exif(self, image_path: str):
        """
        修改后：同时返回用于模型推理的 Tensor 和用于前端展示的原始字典
        """
        import exifread
        import math
        
        exif_features = [0.0] * self.EXIF_DIM
        # 用于前端雷达图展示的原始数据字典
        raw_exif_data = {
            'FocalLength': 0.0,
            'FNumber': 0.0,
            'ExposureTime': 0.0,
            'ISO': 0.0,
            'Brightness': 0.0
        }

        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            if tags:
                def to_float(val, default=0.0):
                    if val is None: return default
                    try:
                        if hasattr(val, 'values'):
                            v = val.values[0]
                            if hasattr(v, 'num') and hasattr(v, 'den'):
                                return float(v.num) / float(v.den) if v.den != 0 else default
                            return float(v)
                        return float(val)
                    except Exception:
                        return default

                # 提取核心参数
                raw_exif_data['FocalLength'] = to_float(tags.get('EXIF FocalLength'))
                raw_exif_data['FNumber'] = to_float(tags.get('EXIF FNumber'))
                raw_exif_data['ExposureTime'] = to_float(tags.get('EXIF ExposureTime'))
                raw_exif_data['ISO'] = to_float(tags.get('EXIF ISOSpeedRatings'))
                raw_exif_data['Brightness'] = to_float(tags.get('EXIF BrightnessValue'))
                
                flash_val = tags.get('EXIF Flash')
                flash_fired = 0.0
                if flash_val and hasattr(flash_val, 'values') and isinstance(flash_val.values[0], int):
                    flash_fired = float(flash_val.values[0] & 1)
                
                time_sin, time_cos = 0.0, 0.0
                datetime_tag = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTimeOriginal')
                if datetime_tag:
                    try:
                        time_part = str(datetime_tag).strip().split(' ')[1]
                        h, m, s = map(int, time_part.split(':'))
                        decimal_hour = h + (m / 60.0) + (s / 3600.0)
                        time_sin = math.sin(2 * math.pi * decimal_hour / 24.0)
                        time_cos = math.cos(2 * math.pi * decimal_hour / 24.0)
                    except Exception:
                        pass

                # 组装 8 维特征张量
                exif_features[0] = raw_exif_data['FocalLength']
                exif_features[1] = raw_exif_data['FNumber']
                exif_features[2] = max(1e-6, min(raw_exif_data['ExposureTime'], 1.0)) if raw_exif_data['ExposureTime'] > 0 else 0.0
                exif_features[3] = raw_exif_data['ISO']
                exif_features[4] = raw_exif_data['Brightness']
                exif_features[5] = flash_fired
                exif_features[6] = time_sin
                exif_features[7] = time_cos

        except Exception as e:
            logger.error(f"读取 EXIF 异常: {e}")

        tensor_features = torch.tensor(exif_features, dtype=torch.float32)
        for i in range(self.EXIF_DIM):
            if self.EXIF_STD[i] > 0 and tensor_features[i] != 0.0:
                tensor_features[i] = (tensor_features[i] - self.EXIF_MEAN[i]) / self.EXIF_STD[i]
                
        return tensor_features.unsqueeze(0).to(self.device), raw_exif_data

    def predict(self, image_path: str) -> dict:
        """
        修改后：返回包含分数和 EXIF 数据的字典
        """
        try:
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 获取张量与前端可视化数据
            exif_tensor, raw_exif_data = self._extract_and_normalize_exif(image_path)
            
            with torch.no_grad():
                output = self.model(img_tensor, exif_tensor)
                score_normalized = output.item()
                
            score_100 = max(0.0, min(100.0, score_normalized * 100.0))
            
            return {
                'score': round(score_100, 2),
                'exif': raw_exif_data
            }
        except Exception as e:
            logger.error(f"预测报错: {e}")
            raise e