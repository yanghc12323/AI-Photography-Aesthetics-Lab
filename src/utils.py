"""
评估指标工具模块
提供图像质量评价(IQA)领域的标准评估指标
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def calculate_metrics(preds, targets):
    """
    计算IQA核心评估指标
    
    参数:
        preds: 预测分数列表或numpy数组
        targets: 真实分数列表或numpy数组
    
    返回:
        dict: 包含PLCC、SRCC、RMSE、MAE的字典
    
    注意:
        - PLCC/SRCC越接近1越好
        - RMSE/MAE越接近0越好
    """
    preds = np.array(preds, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    
    # 基本验证
    if len(preds) != len(targets):
        raise ValueError(f"预测值数量({len(preds)})与真实值数量({len(targets)})不匹配")
    
    if len(preds) == 0:
        raise ValueError("输入数据为空")
    
    # PLCC (Pearson Linear Correlation Coefficient)
    plcc, _ = pearsonr(preds, targets)
    
    # SRCC (Spearman Rank Correlation Coefficient)
    srcc, _ = spearmanr(preds, targets)
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(preds - targets))
    
    return {
        'PLCC': plcc,
        'SRCC': srcc,
        'RMSE': rmse,
        'MAE': mae
    }


def print_metrics(metrics, phase='Validation', logger=None):
    """
    格式化打印评估指标
    
    参数:
        metrics: calculate_metrics返回的字典
        phase: 阶段标识（Training/Validation/Test）
        logger: 可选的logger对象，为None时使用print
    """
    log_func = logger.info if logger else print
    
    log_func(f"\n{'='*60}")
    log_func(f"{phase} 评估结果")
    log_func('='*60)
    log_func(f"PLCC:  {metrics['PLCC']:.6f}  (Pearson线性相关系数，目标: >0.8)")
    log_func(f"SRCC:  {metrics['SRCC']:.6f}  (Spearman秩相关系数，目标: >0.8)")
    log_func(f"RMSE:  {metrics['RMSE']:.6f}  (均方根误差)")
    log_func(f"MAE:   {metrics['MAE']:.6f}  (平均绝对误差)")
    log_func('='*60)

# utils.py - 在文件末尾添加此函数

def plot_training_curves(train_losses, val_losses, val_plccs, val_srccs, save_path):
    """
    绘制训练曲线（Loss、PLCC、SRCC）
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_plccs: 验证 PLCC 列表
        val_srccs: 验证 SRCC 列表
        save_path: 图片保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: Loss 曲线
    axes[0].plot(train_losses, label='Train Loss', linewidth=2, color='blue')
    axes[0].plot(val_losses, label='Val Loss', linewidth=2, color='red')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 图2: PLCC 曲线
    axes[1].plot(val_plccs, label='Val PLCC', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('PLCC', fontsize=12)
    axes[1].set_title('Validation PLCC', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 图3: SRCC 曲线
    axes[2].plot(val_srccs, label='Val SRCC', linewidth=2, color='purple')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('SRCC', fontsize=12)
    axes[2].set_title('Validation SRCC', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()