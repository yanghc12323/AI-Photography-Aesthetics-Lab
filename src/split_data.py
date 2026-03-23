import os
import shutil
import random
from tqdm import tqdm

# ================= 配置路径 =================
SOURCE_DIR = r'E:/SPAQ zip/TestImage' 

TARGET_TRAIN_DIR = '../images/train'
TARGET_VAL_DIR = '../images/val'
# ==========================================

def main():
    # 1. 创建目标文件夹
    os.makedirs(TARGET_TRAIN_DIR, exist_ok=True)
    os.makedirs(TARGET_VAL_DIR, exist_ok=True)

    # 2. 获取所有图片
    print(f"正在读取 {SOURCE_DIR} 下的图片...")
    all_images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"共找到 {len(all_images)} 张图片。")

    if len(all_images) == 0:
        print("❌ 没找到图片，请检查 SOURCE_DIR 路径是否正确！")
        return

    # 3. 随机打乱并按 8:2 划分
    random.seed(42)  # 固定随机种子，保证每次划分结果一样
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # 4. 开始复制文件 (使用 copy 而不是 move，防止你操作失误搞坏原数据)
    print("\n📦 开始构建训练集 (Train)...")
    for img in tqdm(train_images):
        shutil.copy(os.path.join(SOURCE_DIR, img), os.path.join(TARGET_TRAIN_DIR, img))

    print("\n📦 开始构建验证集 (Val)...")
    for img in tqdm(val_images):
        shutil.copy(os.path.join(SOURCE_DIR, img), os.path.join(TARGET_VAL_DIR, img))

    print("\n✅ 数据集划分大功告成！现在可以运行 train.py 了！")

if __name__ == '__main__':
    main()