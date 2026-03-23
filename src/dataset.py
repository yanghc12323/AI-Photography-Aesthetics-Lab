import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# ==================== EXIF 特征预处理 ====================
class EXIFPreprocessor:
    """EXIF 特征预处理类"""
    
    # EXIF 特征维度：8 维（不包括 Image name）
    EXIF_DIM = 8
    
    # 归一化参数（8 个特征）
    # 基于实际数据统计的归一化参数（排除异常值后，2026-03-22 更新）
    EXIF_MEAN = [3.97, 2.09, 0.023, 309.70, 1.50, 0.024, -0.47, -0.57]
    EXIF_STD = [1.27, 0.20, 0.034, 693.45, 3.04, 0.15, 0.52, 0.42]
    
    # Exposure time 截断范围（防止异常值影响）
    EXPOSURE_TIME_MIN = 1e-6   # 最小 1 微秒
    EXPOSURE_TIME_MAX = 1.0    # 最大 1 秒
    
    def __init__(self, excel_path):
        """
        参数:
            excel_path: EXIF 标签 Excel 文件路径
        """
        self.exif_df = pd.read_excel(excel_path)
        
        # 建立图片名到 EXIF 特征的映射
        self.exif_dict = {}
        
        # 使用列索引读取（避免列名不一致问题）
        # 列索引：1-8 对应 8 个 EXIF 特征（0 是 Image name）
        for _, row in self.exif_df.iterrows():
            img_name = row.iloc[0]  # 第 1 列是图片名
            
            # 提取 8 个 EXIF 特征（列索引 1-8）
            try:
                exif_features = [float(row.iloc[i]) for i in range(1, 9)]
                
                # ✅ 截断 Exposure time（第 3 个特征，索引 2）
                exif_features[2] = max(
                    self.EXPOSURE_TIME_MIN,
                    min(exif_features[2], self.EXPOSURE_TIME_MAX)
                )
                
                self.exif_dict[img_name] = torch.tensor(exif_features, dtype=torch.float32)
            except (KeyError, TypeError, ValueError, IndexError) as e:
                # 如果读取失败，使用零向量
                self.exif_dict[img_name] = torch.zeros(self.EXIF_DIM, dtype=torch.float32)
    
    def get_exif_features(self, image_name):
        """
        获取单张图片的 EXIF 特征
        
        参数:
            image_name: 图片文件名
            
        返回:
            torch.Tensor: 归一化后的 EXIF 特征 (8 维)
        """
        if image_name not in self.exif_dict:
            return torch.zeros(self.EXIF_DIM, dtype=torch.float32)
        
        features = self.exif_dict[image_name].clone()
        
        # 归一化
        for i in range(len(features)):
            if self.EXIF_STD[i] > 0:
                features[i] = (features[i] - self.EXIF_MEAN[i]) / self.EXIF_STD[i]
        
        return features


class SPAQDataset(Dataset):
    """SPAQ 数据集加载类"""
    
    def __init__(self, image_dir, excel_path, exif_path=None, transform=None, 
                 return_filename=False, use_exif=False):
        """
        参数:
            image_dir: 图片文件夹路径
            excel_path: MOS 标签 Excel 文件路径
            exif_path: EXIF 标签 Excel 文件路径（可选）
            transform: 图片变换操作
            return_filename: 是否返回文件名
            use_exif: 是否使用 EXIF 特征
        """
        self.image_dir = image_dir
        self.transform = transform
        self.return_filename = return_filename 
        self.use_exif = use_exif
        
        # 读取 Excel 标签文件
        self.labels_df = pd.read_excel(excel_path)
        
        # 只保留需要的列
        self.labels_df = self.labels_df[['Image name', 'MOS']]

        # 加载 EXIF 预处理器
        self.exif_preprocessor = None
        if use_exif and exif_path:
            self.exif_preprocessor = EXIFPreprocessor(exif_path)
            print(f"已加载 EXIF 预处理器")
        
        # 过滤掉图片不存在或损坏的样本
        self.valid_samples = []
        skipped = 0  # 记录跳过的样本数
        for idx, row in self.labels_df.iterrows():
            img_path = os.path.join(image_dir, row['Image name'])
            if not os.path.exists(img_path):
                skipped += 1
                continue
            
            # 尝试验证图片完整性
            try:
                # 打开图片并快速验证（不加载全部像素）
                img = Image.open(img_path)
                img.verify()      # 如果损坏会抛出异常
                img.close()       # 显式关闭文件句柄
                
                # 验证通过，加入有效样本
                self.valid_samples.append({
                    'image_path': img_path,
                    'mos_score': row['MOS'],
                    'image_name': row['Image name']
                })
            except (UnidentifiedImageError, IOError, OSError) as e:
                # 捕获与图片损坏相关的异常，打印警告并跳过
                print(f"跳过损坏图片：{img_path} - {e}")
                skipped += 1
                continue
        
        print(f"✓ 成功加载 {len(self.valid_samples)} 个有效样本 (跳过 {skipped} 个)")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回:
            如果 return_filename=True: (image, target, filename)
            如果 return_filename=False: (image, target)
        """
        # 获取样本信息（此时保证图片可读）
        sample = self.valid_samples[idx]
        image_path = sample['image_path']
        mos_score = sample['mos_score']
        image_name = sample['image_name']
        
        # 加载图片（不会再抛出 UnidentifiedImageError）
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # MOS 分数归一化到 0-1 之间（SPAQ 的 MOS 范围是 0-100）
        mos_normalized = mos_score / 100.0
        target = torch.tensor(mos_normalized, dtype=torch.float32)
        
        # 构建返回值
        if self.use_exif:
            exif_features = self.exif_preprocessor.get_exif_features(image_name)
            if self.return_filename:
                return image, exif_features, target, image_name
            else:
                return image, exif_features, target
        else:
            if self.return_filename:
                return image, target, image_name
            else:
                return image, target


def get_transform(phase='train'):
    """
    获取数据变换
    参数:
        phase: 'train' 或 'val'
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ==================== 新增：数据集划分功能 ====================
def split_dataset(image_dir='E:/【PROJECT】SPAQ/images/origin/TestImage',  # ✅ 修改为 origin 文件夹
                  excel_path='E:/【PROJECT】SPAQ/data/MOS and Image attribute scores.xlsx',
                  output_dir='E:/【PROJECT】SPAQ/images',
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                  random_seed=42):
    """
    数据集划分函数：将原始数据按指定比例划分为 train/val/test
    
    参数:
        image_dir: 原始图片文件夹路径（已修改为 origin）
        excel_path: MOS 标签 Excel 文件路径
        output_dir: 输出文件夹路径（划分后的 train/val/test 子文件夹）
        train_ratio: 训练集比例（默认 0.7）
        val_ratio: 验证集比例（默认 0.2）
        test_ratio: 测试集比例（默认 0.1）
        random_seed: 随机种子（保证可复现）
    
    返回:
        dict: 包含 train/val/test 图片名列表的字典
    """
    import shutil
    from sklearn.model_selection import train_test_split
    import json
    
    print("=" * 60)
    print("📊 SPAQ 数据集划分工具")
    print("=" * 60)
    
    # 验证比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1"
    
    # 路径检查
    print("\n🔍 检查路径配置...")
    print(f"   原始图片目录：{image_dir}")
    print(f"   Excel 文件路径：{excel_path}")
    print(f"   输出目录：{output_dir}")
    
    if not os.path.exists(excel_path):
        print(f"\n❌ 错误：Excel 文件不存在！")
        return None
    
    if not os.path.exists(image_dir):
        print(f"\n❌ 错误：原始图片目录不存在！")
        print(f"   路径：{image_dir}")
        return None
    print(f"   ✓ 路径检查通过")
    
    # 读取标签文件
    print("\n📂 读取标签文件...")
    df = pd.read_excel(excel_path)
    image_names = df['Image name'].tolist()
    total_images = len(image_names)
    print(f"   Excel 中图片数量：{total_images}")
    
    # 第一次划分：train vs (val+test)
    print(f"\n🔄 第一次划分：训练集 ({train_ratio*100:.0f}%) vs (验证集 + 测试集)...")
    train_images, temp_images = train_test_split(
        image_names, 
        test_size=(1 - train_ratio),
        random_state=random_seed
    )
    print(f"   训练集：{len(train_images)} 张 ({len(train_images)/total_images*100:.1f}%)")
    
    # 第二次划分：val vs test
    print(f"\n🔄 第二次划分：验证集 ({val_ratio*100:.0f}%) vs 测试集 ({test_ratio*100:.0f}%)...")
    val_images, test_images = train_test_split(
        temp_images,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed
    )
    print(f"   验证集：{len(val_images)} 张 ({len(val_images)/total_images*100:.1f}%)")
    print(f"   测试集：{len(test_images)} 张 ({len(test_images)/total_images*100:.1f}%)")
    
    # 创建文件夹
    print("\n📁 创建文件夹...")
    for split in ['train', 'val', 'test']:
        folder_path = os.path.join(output_dir, split)
        os.makedirs(folder_path, exist_ok=True)
        print(f"   ✓ {folder_path}")
    
    # 复制图片
    def copy_images(images, split_name):
        """复制图片到目标文件夹"""
        success = 0
        failed = 0
        failed_samples = []
        for img_name in images:
            src = os.path.join(image_dir, img_name)
            dst = os.path.join(output_dir, split_name, img_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                success += 1
            else:
                failed += 1
                if len(failed_samples) < 5:
                    failed_samples.append(img_name)
        if failed_samples:
            print(f"   ⚠️  失败示例：{failed_samples}")
        return success, failed
    
    print("\n📋 复制训练集图片...")
    success, failed = copy_images(train_images, 'train')
    print(f"   ✓ 成功：{success}, 失败：{failed}")
    
    print("\n📋 复制验证集图片...")
    success, failed = copy_images(val_images, 'val')
    print(f"   ✓ 成功：{success}, 失败：{failed}")
    
    print("\n📋 复制测试集图片...")
    success, failed = copy_images(test_images, 'test')
    print(f"   ✓ 成功：{success}, 失败：{failed}")
    
    # 保存划分信息
    print("\n💾 保存划分信息...")
    split_info = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    with open('E:/【PROJECT】SPAQ/data/split_info.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    print("   ✓ split_info.json")
    
    print("\n" + "=" * 60)
    print("✅ 数据划分完成！")
    print("=" * 60)
    print("\n⚠️  下一步操作：")
    print("   1. 修改 train.py 中的 image_dir 为 'train' 子文件夹")
    print("   2. 修改 train.py 中的 val_dir 为 'val' 子文件夹")
    print("   3. 修改 test.py 中的 image_dir 为 'test' 子文件夹")
    print("   4. 重新运行 train.py 训练模型")
    print("   5. 运行 test.py 进行测试")
    print("=" * 60)
    
    return split_info


# ==================== 主程序入口（可选） ====================
if __name__ == '__main__':
    # 取消下面注释即可运行数据划分
    split_dataset(
        train_ratio=0.7,  # 70% 训练集
        val_ratio=0.2,    # 20% 验证集
        test_ratio=0.1,   # 10% 测试集
        random_seed=42
    )