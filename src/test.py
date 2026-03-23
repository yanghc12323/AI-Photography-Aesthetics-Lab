"""
SPAQ模型测试脚本
加载最佳模型并在测试集上评估性能
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from dataset import SPAQDataset, get_transform
from model import create_model
from utils import calculate_metrics, print_metrics


# ==================== 配置参数 ====================
class TestConfig:
    # 数据路径
    IMAGE_DIR = 'E:/【PROJECT】SPAQ/images'
    EXCEL_PATH = 'E:/【PROJECT】SPAQ/data/MOS and Image attribute scores.xlsx'
    EXIF_PATH = 'E:/【PROJECT】SPAQ/data/EXIF_tags.xlsx'
    
    # 测试参数
    BATCH_SIZE = 32
    NUM_WORKERS = 1
    
    # EXIF特征配置
    USE_EXIF = True

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型路径
    MODEL_PATH = 'E:/【PROJECT】SPAQ/checkpoints/best_model.pth'
    
    # 保存路径
    SAVE_DIR = 'E:/【PROJECT】SPAQ/test_results'


def load_checkpoint(model, checkpoint_path, device):
    """
    加载模型检查点
    
    返回:
        模型、检查点字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def test_model(model, dataloader, criterion, device, use_exif=False):
    """
    测试模型并计算评估指标
    
    返回:
        tuple: (平均损失, 预测列表, 真实列表, 指标字典)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_image_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            if use_exif:
                images, exif_features, targets, image_names = batch
                exif_features = exif_features.to(device)
            else:
                images, targets, image_names = batch
            
            images = images.to(device)
            targets = targets.to(device)
            
            if use_exif:
                outputs = model(images, exif_features)
            else:
                outputs = model(images)
            
            loss = criterion(outputs.squeeze(1), targets)
            
            total_loss += loss.item()
            all_preds.extend(outputs.squeeze(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_image_names.extend(image_names)
    
    metrics = calculate_metrics(all_preds, all_targets)
    
    return total_loss / len(dataloader), all_preds, all_targets, all_image_names, metrics


def save_predictions(preds, targets, image_names, save_path):
    """保存预测结果到CSV文件"""
    import csv
    
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_Name', 'Predicted_Score', 'True_Score', 'Difference'])
        
        for name, pred, target in zip(image_names, preds, targets):
            writer.writerow([name, f'{pred:.6f}', f'{target:.6f}', f'{pred - target:.6f}'])


def main():
    # 配置日志
    os.makedirs(TestConfig.SAVE_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(TestConfig.SAVE_DIR, 'test.log'), mode='w', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 使用设备：{TestConfig.DEVICE}")
    logger.info(f"📂 模型路径：{TestConfig.MODEL_PATH}")
    
    # ==================== 1. 加载模型 ====================
    logger.info("\n🧠 加载模型...")
    
    if not os.path.exists(TestConfig.MODEL_PATH):
        logger.error(f"❌ 模型文件不存在：{TestConfig.MODEL_PATH}")
        logger.error("请先运行 train.py 训练模型")
        return
    
    model = create_model(pretrained=False, freeze_backbone=False, use_exif=TestConfig.USE_EXIF)
    model, checkpoint = load_checkpoint(model, TestConfig.MODEL_PATH, TestConfig.DEVICE)
    model = model.to(TestConfig.DEVICE)
    model.eval()
    
    # 打印模型信息
    logger.info(f"✅ 模型加载成功")
    logger.info(f"📊 训练时的最佳验证PLCC: {checkpoint.get('val_plcc', 'N/A'):.6f}")
    logger.info(f"📊 训练时的最佳验证SRCC: {checkpoint.get('val_srcc', 'N/A'):.6f}")
    logger.info(f"📊 保存时的Epoch: {checkpoint.get('epoch', 'N/A') + 1}")
    
    # ==================== 2. 加载测试数据 ====================
    logger.info("\n📊 加载测试数据集...")
    
    # 使用验证集作为测试集（如果你有独立的测试集，可以修改路径）
    test_dataset = SPAQDataset(
        image_dir=os.path.join(TestConfig.IMAGE_DIR, 'test'),
        excel_path=TestConfig.EXCEL_PATH,
        exif_path=TestConfig.EXIF_PATH if TestConfig.USE_EXIF else None,
        transform=get_transform('val'),
        return_filename=True,
        use_exif=TestConfig.USE_EXIF 
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TestConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=TestConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    logger.info(f"测试集样本数：{len(test_dataset)}")
    
    # ==================== 3. 测试评估 ====================
    logger.info("\n开始测试...")
    
    criterion = nn.MSELoss()
    
    test_loss, preds, targets, image_names, metrics = test_model(
        model, test_loader, criterion, TestConfig.DEVICE, use_exif=TestConfig.USE_EXIF
    )
    
    # 打印测试结果
    print_metrics(metrics, phase='Test', logger=logger)
    
    logger.info(f"\n📊 测试损失：{test_loss:.6f}")
    
    # ==================== 4. 保存预测结果 ====================
    logger.info("\n💾 保存预测结果...")
    
    predictions_path = os.path.join(TestConfig.SAVE_DIR, 'predictions.csv')
    save_predictions(preds, targets, image_names, predictions_path)
    logger.info(f"✓ 预测结果已保存至：{predictions_path}")
    
    # ==================== 5. 分析预测分布 ====================
    logger.info("\n📈 预测分布分析...")
    
    import numpy as np
    preds_np = np.array(preds)
    targets_np = np.array(targets)
    
    logger.info(f"预测值 - 均值：{preds_np.mean():.4f}, 标准差：{preds_np.std():.4f}")
    logger.info(f"真实值 - 均值：{targets_np.mean():.4f}, 标准差：{targets_np.std():.4f}")
    logger.info(f"预测值范围：[{preds_np.min():.4f}, {preds_np.max():.4f}]")
    logger.info(f"真实值范围：[{targets_np.min():.4f}, {targets_np.max():.4f}]")
    
    # ==================== 6. 测试总结 ====================
    logger.info(f"\n{'='*60}")
    logger.info("✅ 测试完成！")
    logger.info('='*60)
    logger.info(f"最佳PLCC: {metrics['PLCC']:.6f}")
    logger.info(f"最佳SRCC: {metrics['SRCC']:.6f}")
    logger.info(f"结果保存路径：{TestConfig.SAVE_DIR}")
    logger.info('='*60)


if __name__ == '__main__':
    main()