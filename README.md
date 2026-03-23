# 📸 AI Photography Aesthetics Lab | 摄影美学多模态交互评测系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Backend-000000.svg)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-UI-38B2AC.svg)

本项目是一个端到端的多模态图像美学质量评估（IQA）系统，致力于解决传统深度学习模型在摄影应用中的“主观感知差异”与“评价不可解释性”痛点。

系统不仅在底层算法上创新性地融合了 **ResNet50 视觉特征**与 **8 维 EXIF 物理先验特征**，更在前端产品层构建了包含“人机感知盲测”、“What-If 参数优化”和“并发 A/B 评测室”的完整人机交互（HCI）闭环。

---

## ✨ 核心特性 (Core Features)

- 🧠 **多模态特征融合网络 (SPAQRegressor)**：跳出纯像素限制，深度融合光圈、快门、ISO 及时间周期拓扑特征，精准拟合人类摄影专家的光学感知。
- ⚖️ **人机感知盲测交互**：强制引入主观感知环节，系统根据“人机分数差”动态生成差异化语义评语，探讨情感叙事与硬性技术指标的认知鸿沟。
- 💡 **可解释性 AI (What-If Engine)**：基于底层二进制 EXIF 解析与专家规则库，动态推演曝光参数优化策略（如防抖、降噪、景深控制）。
- ⚔️ **高可用 A/B 对比评测室**：专为修图与算法验证设计，后端引入 UUID 绝对隔离机制，完美解决高并发下的文件竞态与存储泄漏问题，支持最高 32MB 超清原图解析。

---
### 🚀 性能评测对比 (Performance on SPAQ Dataset)

本系统在极轻量级的推理开销下（适合 Web 实时响应），取得了逼近前沿 SOTA 模型的卓越主客观对齐精度。

| 模型名称 (Model) | 核心算法架构 | PLCC $\uparrow$ | SRCC $\uparrow$ | 部署体量 |
| :--- | :--- | :---: | :---: | :---: |
| BRISQUE (TIP 2012) | 传统 NSS + SVR | ~0.805 | ~0.801 | 极轻量 |
| NIMA (TIP 2018) | VGG/ResNet 分布预测 | 0.907 | 0.910 | 轻量 |
| **SPAQRegressor (Ours)**| **ResNet50 + EXIF 物理多模态融合** | **0.908** | **0.899** | **轻量** |
| MUSIQ (ICCV 2021) | 多尺度切割 + Vision Transformer | 0.924 | 0.929 | 极重 |

*\*注：本模型在测试集上的真实表现为 PLCC 0.9078, SRCC 0.8987, RMSE 0.0881。*

---


## 🛠️ 快速开始 (Quick Start)

### 1. 克隆与环境依赖
```bash
git clone [https://github.com/YourUsername/YourRepository.git](https://github.com/YourUsername/YourRepository.git)
cd YourRepository
pip install torch torchvision pandas Pillow flask werkzeug exifread echarts
```

### 2. 启动 Web 服务
```bash
python app.py
#服务启动后，在浏览器中访问 http://127.0.0.1:5000 即可体验完整的交互式评测实验室。
```
---
###📁 核心目录结构
```Plaintext
.
├── app.py                 # Flask 后端路由与并发控制核心
├── inference.py           # 独立的多模态端到端推理管道（底层依赖 exifread 二进制解析）
├── model.py               # SPAQRegressor 网络架构定义
├── dataset.py             # 数据预处理与 EXIF 特征清洗模块
├── train.py / test.py     # 模型训练回路与测试评估脚本
├── checkpoints/
│   └── best_model.pth     # 训练收敛的最佳模型权重
└── templates/
    └── index.html         # 基于 Tailwind + 原生 JS 的响应式交互前端
```
---
### 👨‍💻 作者信息
杨皓臣 (Haochen Yang) 
清华大学 (Tsinghua University) - 未央书院 (Weiyang College) 
研究兴趣：人机交互 (HCI)、用户体验设计 (UX)、深度学习在视觉评估中的应用
