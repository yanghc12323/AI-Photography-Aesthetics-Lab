"""
SPAQ图像质量评分模型训练脚本
支持迁移学习、早停策略、多指标评估
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import argparse
import json 

from dataset import SPAQDataset, get_transform
from model import create_model
from utils import calculate_metrics, print_metrics, plot_training_curves

# ==================== 配置参数 ====================
class Config:
    # 数据路径
    IMAGE_DIR = 'E:/【PROJECT】SPAQ/images'
    EXCEL_PATH = 'E:/【PROJECT】SPAQ/data/MOS and Image attribute scores.xlsx'
    EXIF_PATH = 'E:/【PROJECT】SPAQ/data/EXIF_tags.xlsx' 
    
    # 训练参数
    BATCH_SIZE = 64
    NUM_EPOCHS = 60
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4  # 权重衰减，防止过拟合
    NUM_WORKERS = 4
    
    # EXIF特征配置
    USE_EXIF = True

    # 早停策略参数
    EARLY_STOP_PATIENCE = 20  # 容忍多少个epoch无改善
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 保存路径
    SAVE_DIR = 'E:/【PROJECT】SPAQ/checkpoints'
    
    # 随机种子（保证可复现性）
    RANDOM_SEED = 42


def set_random_seed(seed):
    """设置随机种子，保证实验可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    加载模型检查点
    
    参数:
        checkpoint_path: checkpoint文件路径
        model: 模型对象
        optimizer: 优化器对象
        device: 设备
        
    返回:
        dict: 包含epoch、指标等信息的字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在：{checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 返回其他信息
    return {
        'epoch': checkpoint['epoch'],
        'val_plcc': checkpoint.get('val_plcc', 0),
        'val_srcc': checkpoint.get('val_srcc', 0),
        'val_rmse': checkpoint.get('val_rmse', 0),
        'val_mae': checkpoint.get('val_mae', 0),
    }

def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False):
    """
    保存模型检查点
    
    参数:
        is_best: 是否为最佳模型
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_plcc': metrics['PLCC'],
        'val_srcc': metrics['SRCC'],
        'val_rmse': metrics['RMSE'],
        'val_mae': metrics['MAE'],
    }
    
    filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
    torch.save(checkpoint, os.path.join(save_path, filename))


def save_training_history(train_losses, val_losses, val_plccs, val_srccs, 
                          val_rmses, val_maes, save_path):
    """
    保存训练历史到 JSON 文件
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_plccs: 验证 PLCC 列表
        val_srccs: 验证 SRCC 列表
        val_rmses: 验证 RMSE 列表
        val_maes: 验证 MAE 列表
        save_path: 保存路径
    """
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_plccs': val_plccs,
        'val_srccs': val_srccs,
        'val_rmses': val_rmses,
        'val_maes': val_maes,
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def load_training_history(load_path):
    """
    加载训练历史（兼容旧版本）
    
    参数:
        load_path: 历史文件路径
        
    返回:
        dict: 训练历史字典（包含所有指标）
    """
    if not os.path.exists(load_path):
        return None
    with open(load_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    # 兼容旧版本：如果缺少新字段，初始化为空列表
    if 'val_srccs' not in history:
        history['val_srccs'] = []
    if 'val_rmses' not in history:
        history['val_rmses'] = []
    if 'val_maes' not in history:
        history['val_maes'] = []
    
    return history
    
def train_one_epoch(model, dataloader, criterion, optimizer, device, use_exif=False):
    """
    训练一个epoch
    
    返回:
        float: 平均训练损失
    """
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress_bar:
        if use_exif:
            images, exif_features, targets = batch
            exif_features = exif_features.to(device)
        else:
            images, targets = batch
        
        images = images.to(device)
        targets = targets.to(device)
        
        # 前向传播
        if use_exif:
            outputs = model(images, exif_features)
        else:
            outputs = model(images)
        
        loss = criterion(outputs.squeeze(1), targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, use_exif=False):
    """
    验证并计算评估指标
    
    返回:
        tuple: (平均损失, 预测列表, 真实列表, 指标字典)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            if use_exif:
                images, exif_features, targets = batch
                exif_features = exif_features.to(device)
            else:
                images, targets = batch
            
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
    
    metrics = calculate_metrics(all_preds, all_targets)
    
    return total_loss / len(dataloader), all_preds, all_targets, metrics

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SPAQ模型训练')
    
    parser.add_argument('--resume', type=str, default=None,
                        help='续训的checkpoint路径（例如：checkpoints/checkpoint_epoch_20.pth）')
    parser.add_argument('--epochs', type=int, default=None,
                        help='总训练轮次（续训时可覆盖默认值）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（续训时可覆盖默认值）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（续训时可覆盖默认值）')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    # 设置随机种子
    set_random_seed(Config.RANDOM_SEED)
    
    # 配置日志
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # 根据是否续训决定日志模式
    log_mode = 'a' if args.resume else 'w'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(Config.SAVE_DIR, 'training.log'), 
                              mode=log_mode, encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 覆盖配置参数（如果命令行指定）
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.lr:
        Config.LEARNING_RATE = args.lr
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    
    logger.info(f"🚀 使用设备：{Config.DEVICE}")
    logger.info(f"📋 配置参数：BatchSize={Config.BATCH_SIZE}, LR={Config.LEARNING_RATE}, WeightDecay={Config.WEIGHT_DECAY}")
    
    # 创建保存目录
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # ==================== 1. 加载数据 ====================
    logger.info("\n📊 加载数据集...")
    
    train_dataset = SPAQDataset(
        image_dir=os.path.join(Config.IMAGE_DIR, 'train'),
        excel_path=Config.EXCEL_PATH,
        exif_path=Config.EXIF_PATH if Config.USE_EXIF else None,
        transform=get_transform('train'),
        use_exif=Config.USE_EXIF
    )
    
    val_dataset = SPAQDataset(
        image_dir=os.path.join(Config.IMAGE_DIR, 'val'),
        excel_path=Config.EXCEL_PATH,
        exif_path=Config.EXIF_PATH if Config.USE_EXIF else None,
        transform=get_transform('val'),
        use_exif=Config.USE_EXIF
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True  # 避免batchnorm问题
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # ==================== 2. 创建模型 ====================
    logger.info("\n🧠 创建模型...")
    model = create_model(
        pretrained=True, 
        freeze_backbone=False,
        use_exif=Config.USE_EXIF
    )
    model = model.to(Config.DEVICE)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量：{total_params:,} | 可训练参数量：{trainable_params:,}")
    
    # ==================== 3. 定义损失和优化器 ====================
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY  # 添加权重衰减
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # ==================== 4. 加载checkpoint（如果续训） ====================
    start_epoch = 0
    best_val_plcc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_plccs = []
    val_srccs = [] 
    val_rmses = []
    val_maes = [] 
    
    if args.resume:
        logger.info(f"\n加载checkpoint续训：{args.resume}")
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, Config.DEVICE)
        start_epoch = checkpoint_info['epoch'] + 1
        best_val_plcc = checkpoint_info.get('val_plcc', 0)
        
        # 加载训练历史
        history = load_training_history(os.path.join(Config.SAVE_DIR, 'training_history.json'))
        if history:
            train_losses = history['train_losses']
            val_losses = history['val_losses']
            val_plccs = history['val_plccs']
            val_srccs = history['val_srccs']
            val_rmses = history['val_rmses']  
            val_maes = history['val_maes']   
            logger.info(f"已恢复训练历史（{len(train_losses)}个 epoch）")
        
        logger.info(f"从Epoch {start_epoch}继续训练，当前最佳PLCC={best_val_plcc:.6f}")
        
        # 调整学习率调度器状态
        for _ in range(len(val_losses)):
            scheduler.step(val_losses[_] if _ < len(val_losses) else val_losses[-1])
    else:
        logger.info("\n从头开始训练")

    # ==================== 5. 训练循环 ====================
    logger.info("\n开始训练...")
    
    early_stop_triggered = False
    
    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        logger.info('='*60)
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, use_exif=Config.USE_EXIF
        )
        train_losses.append(train_loss)
        
        # 计算训练集PLCC（需要再跑一遍训练集，简化起见用最近batch估计）
        # 完整实现可以单独写一个evaluate函数
        
        # 验证
        val_loss, val_preds, val_targets, val_metrics = validate(
            model, val_loader, criterion, Config.DEVICE, use_exif=Config.USE_EXIF
        )
        val_losses.append(val_loss)
        val_plccs.append(val_metrics['PLCC'])
        val_srccs.append(val_metrics['SRCC'])
        val_rmses.append(val_metrics['RMSE'])
        val_maes.append(val_metrics['MAE'])

        
        # 打印验证指标
        print_metrics(val_metrics, phase='Validation', logger=logger)
        
        # 学习率调整
        scheduler.step(val_loss)

        # 手动打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型（基于PLCC而不是Loss）
        if val_metrics['PLCC'] > best_val_plcc:
            best_val_plcc = val_metrics['PLCC']
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, Config.SAVE_DIR, is_best=True)
            logger.info(f"保存最佳模型 (PLCC={val_metrics['PLCC']:.6f})")
        else:
            patience_counter += 1
            logger.info(f"PLCC未改善，耐心计数：{patience_counter}/{Config.EARLY_STOP_PATIENCE}")
        
        # 早停检查
        if patience_counter >= Config.EARLY_STOP_PATIENCE:
            logger.info(f"\n早停触发 (连续{Config.EARLY_STOP_PATIENCE}个epoch PLCC无改善)")
            early_stop_triggered = True
            break
        
        # 定期保存检查点（每10个epoch）
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, Config.SAVE_DIR, is_best=False)
        
        # 保存训练历史（每个epoch后都保存，方便随时中断）
        save_training_history(
            train_losses, val_losses, 
            val_plccs, val_srccs, val_rmses, val_maes,
            os.path.join(Config.SAVE_DIR, 'training_history.json')
        )
        
        logger.info(f"📊 摘要 | Train Loss: {train_loss:.6f} | Val PLCC: {val_metrics['PLCC']:.6f} | Val SRCC: {val_metrics['SRCC']:.6f}")
    
    # ==================== 6. 绘制训练曲线 ====================
    logger.info("\n📈 绘制训练曲线...")
    plot_training_curves(
        train_losses, val_losses, 
        val_plccs, val_srccs,
        os.path.join(Config.SAVE_DIR, 'training_curves.png')
    )
    
    # ==================== 7. 训练总结 ====================
    logger.info(f"\n{'='*60}")
    logger.info("✅ 训练完成！")
    logger.info('='*60)
    logger.info(f"总训练轮次：{len(train_losses)}")
    logger.info(f"最佳验证PLCC：{best_val_plcc:.6f}")
    logger.info(f"早停触发：{early_stop_triggered}")
    logger.info(f"模型保存路径：{Config.SAVE_DIR}/best_model.pth")
    logger.info('='*60)


if __name__ == '__main__':
    main()