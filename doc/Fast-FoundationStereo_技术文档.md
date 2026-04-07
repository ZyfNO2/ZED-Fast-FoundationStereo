# Fast-FoundationStereo 技术文档

## 概述

**Fast-FoundationStereo** 是 NVIDIA 于 2026 年 CVPR 发表的**实时零样本立体匹配模型**，基于 Transformer 和 CNN 的混合架构，能够从双目立体图像对中估计每个像素的视差（Disparity），进而转换为深度图用于 3D 重建。

- **论文**: [Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching](https://arxiv.org/abs/2512.11130)
- **参数量**: 14.6M
- **推理引擎**: TensorRT / PyTorch
- **GitHub**: https://github.com/NVLabs/Fast-FoundationStereo

---

## 整体处理流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Fast-FoundationStereo Pipeline                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   输入: 左目图像 (RGB) + 右目图像 (RGB)                          │
│         ↓                                                         │
│   ┌───────────────────────────────────────┐                       │
│   │  1. 图像预处理 (Image Preprocessing)    │                       │
│   │     - ImageNet 归一化 (mean/std)      │                       │
│   │     - InputPadder 填充至32的倍数       │                       │
│   └──────────────────┬────────────────────┘                       │
│                      ↓                                             │
│   ┌───────────────────────────────────────┐                       │
│   │  2. 特征提取 (Feature Extraction)      │  ← EdgeNeXt (timm)   │
│   │     - Stem: 初始卷积层               │                       │
│   │     - Stages[0~3]: 4级特征金字塔     │                       │
│   │       x4 (1/4), x8 (1/8)            │                       │
│   │       x16(1/16), x32(1/32)          │                       │
│   │     - 反卷积上采样 + Skip Connection│                       │
│   └──────────────────┬────────────────────┘                       │
│                      ↓                                             │
│   ┌───────────────────────────────────────┐                       │
│   │  3. 代价体构建 (Cost Volume Build)    │                       │
│   │     - GWC Volume (Group-wise Corr.)  │                       │
│   │     - Concat Volume                  │                       │
│   │     - CorrStem: 3D卷积处理           │                       │
│   └──────────────────┬────────────────────┘                       │
│                      ↓                                             │
│   ┌───────────────────────────────────────┐                       │
│   │  4. 代价聚合 (Cost Aggregation)        │  ← Hourglass 网络    │
│   │     - 3级下采样 (stride=2)           │                       │
│   │     - Feature Attention 融合         │                       │
│   │     - 3级上采样 + Skip Connection    │                       │
│   │     - DisparityAttention Transformer│                       │
│   └──────────────────┬────────────────────┘                       │
│                      ↓                                             │
│   ┌───────────────────────────────────────┐                       │
│   │  5. 初始视差估计 (Initial Disp)       │                       │
│   │     - Classifier: 3D→1              │                       │
│   │     - Softmax + Regression          │                       │
│   └──────────────────┬────────────────────┘                       │
│                      ↓                                             │
│   ┌───────────────────────────────────────┐                       │
│   │  6. GRU迭代优化 (GRU Refinement)      │  ← ConvGRU × N层     │
│   │     - Context Network (cnet)         │                       │
│   │     - Spatial Attention (sam)        │                       │
│   │     - Channel Attention (cam)        │                       │
│   │     - Geo Encoding Volume          │                       │
│   │     - Selective Multi-Update Block  │                       │
│   │     迭代 iters 次 (默认12次)        │                       │
│   └──────────────────┬────────────────────┘                       │
│                      ↓                                             │
│   ┌───────────────────────────────────────┐                       │
│   │  7. 视差上采样 (Upsampling)            │                       │
│   │     - Spatial GRU 上采样到原分辨率    │                       │
│   │     - Context Upsample              │                       │
│   └──────────────────┬────────────────────┘                       │
│                      ↓                                             │
│   输出: 视差图 (Disparity Map, H×W)                               │
│         ↓                                                             │
│   深度 = fx * baseline / disparity                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块详解

### 1. 特征提取器 (Feature Extractor)

**架构**: EdgeNeXt-Small (通过 `timm` 加载)

| 阶段 | 分辨率 | 通道数 | 说明 |
|------|--------|--------|------|
| Stem | H/4, W/4 | 32 | 初始卷积 |
| Stage0 | H/4, W/4 | 48+vit_dim | 浅层特征 |
| Stage1 | H/8, W/8 | 192 | 中层特征 |
| Stage2 | H/16, W/16 | 320 | 深层特征 |
| Stage3 | H/32, W/32 | 608 | 最深层特征 |

**特点**: 使用 DepthAnything 的 ViT 特征增强，多尺度特征融合

### 2. 代价体 (Cost Volume)

两种代价体构建方式:

- **GWC (Group-Wise Correlation)**: 分组相关计算，减少计算量
- **Concat Volume**: 拼接左右特征，保留更多原始信息

优化选项:
- `pytorch1`: PyTorch 原生实现（兼容性好）
- `triton`: Triton GPU kernel 加速（需要额外依赖）

### 3. 代价聚合网络 (Hourglass)

**沙漏结构** (Encoder-Decoder):

```
输入 → Conv(stride=2) → Conv(stride=2) → Conv(stride=2)
  ↓         ↓                    ↓
  ↑    DeConv(cat) ← Deconv(cat)
  ↓
FeatureAtt + DisparityAttention(Transformer)
  ↓
输出
```

### 4. GRU 迭代优化 (Refinement)

**核心创新点**: 使用 ConvGRU 迭代优化视差估计

- **Context Network**: 生成隐藏状态和门控信号
- **Spatial Attention**: 空间注意力聚焦重要区域
- **Channel Attention**: 通道注意力选择有效特征
- **Geo Encoding Volume**: 几何编码代价体提供位置先验
- **Selective Multi-Update Block**: 选择性更新策略

默认迭代次数: 训练时 12 次, 推理时可配置 (`valid_iters`)

### 5. 层级推理 (Hierarchical Inference)

`run_hierachical()` 方法支持粗到精推理:

1. 先在 0.5x 分辨率运行完整推理 → 得到粗糙视差
2. 上采样粗糙视差作为初始值
3. 在全分辨率运行精化迭代

**优势**: 大幅提升速度，精度损失小

---

## 技术栈

### 深度学习框架

| 组件 | 技术 | 版本要求 |
|------|------|---------|
| 核心框架 | PyTorch | ≥ 2.0 |
| 模型库 | timm (PyTorch Image Models) | 最新版 |
| 推理加速 | TensorRT (可选) | 8.x+ |
| GPU 编程 | Triton (可选) | 用于 GWC 加速 |

### 关键依赖

```python
# requirements.txt 核心依赖
torch, torchvision
timm                              # EdgeNeXt 预训练模型
einops                            # 张量操作
omegaconf                        # 配置管理
scipy                             # 科学计算
numpy                            # 数组运算
scikit-image                     # 图像处理
opencv-contrib-python             # OpenCV 扩展
imageio                          # 图像 I/O
pyyaml                           # YAML 配置
open3d                            # 点云处理 (后处理用)
```

### ZED SDK (本项目集成)

| 组件 | 用途 |
|------|------|
| pyzed | SVO 文件读取 |
| Positional Tracking | 位姿矩阵获取 |
| Camera Calibration | 内参、基线获取 |

### 后处理技术栈 (本项目扩展)

| 技术 | 用途 |
|------|------|
| Open3D | 点云滤波 (统计/半径/DBSCAN) |
| NumPy | 数值计算、坐标转换 |
| OpenCV | 图像缩放、边缘检测 |

---

## 数据流与坐标系统

### 立体视觉基础公式

$$
Z = \frac{f_x \cdot baseline}{disparity}$$

其中:
- $Z$: 深度 (米)
- $f_x$: 相机焦距 (像素)
- $baseline$: 双目基线 (米)
- $disparity$: 视差 (像素)

### 坐标系转换

```
OpenCV 坐标系 (depth2xyzmap 输出):
  X → 右, Y → 下, Z → 前
       ↓ cv2zed 转换矩阵
ZED RIGHT_HANDED_Y_UP 坐标系:
  X → 右, Y → 上, Z → 后
       ↓ pose_4x4 变换
世界坐标系:
  多帧融合后的统一坐标系
```

---

## 训练与数据集

### 训练数据

| 数据集 | 类型 | 规模 | 用途 |
|--------|------|------|------|
| 合成数据集 | Synthetic | 140万对 | 主要训练源 |
| Stereo4D | Real | 外部补充 | 泛化性 |

### 评测基准

| 基准 | 场景 | 特点 |
|------|------|------|
| Middlebury | 室内高分辨率 | 经典立体匹配基准 |
| ETH3D | 室内外多视角 | 3D重建基准 |
| KITTI | 自动驾驶 | 真实城市场景 |

### 训练参数

- **优化器**: AdamW
- **学习率**: 1e-3 (cosine decay)
- **Batch Size**: 8 (3090 GPU)
- **混合精度**: AMP (FP16)
- **训练时长**: ~3天 (3090)

---

## 性能指标

### 速度

| 配置 | 分辨率 | FPS | 备注 |
|------|--------|-----|------|
| PyTorch Eager | 1024×512 | ~15 | 兼容模式 |
| TensorRT | 1024×512 | ~60+ | 生产部署 |
| Hierarchical | 1024×512 | ~40+ | 粗到精 |

### 精度 (Middlebury)

| 指标 | Fast-FoundationStereo | FoundationStereo | CREStereo |
|------|----------------------|-------------------|-----------|
| D1-all | < 5% | < 5% | ~7% |
| Bad 2.0 | < 10% | < 10% | ~15% |

---

## 本项目集成方式

### svo_to_ply.py 流程

```
SVO 文件
  ↓ zed SDK
左目/右目图像 + 内参 + 基线 + 位姿
  ↓ FastFoundationStereo.infer()
视差图 → 深度图 → XYZ点云(相机坐标系)
  ↓ cv2zed + pose_4x4
XYZ点云(世界坐标系)
  ↓ 多级滤波
最终 PLY 点云文件
```

### 滤波管线

```
原始点云
  ↓ ① 稀疏空间滤波 (min_pts/bin ≥ 3)
  ↓ ② 时序双向一致性滤波 (前后各≥N帧)
  ↓ ③ Voxel 下采样 (可选)
  ↓ ④ 统计离群点移除 (k=100, std=0.2)
  ↓ ⑤ 半径离群点移除 (r=0.05, nb=10)
  ↓ ⑥ DBSCAN 聚类 (eps=0.08, min_pts=15)
  ↓ ⑦ 锥形放射检测去除
最终干净点云
```

---

## 文件结构说明

```
Fast-FoundationStereo/
├── core/
│   ├── foundation_stereo.py   # 主模型定义 (FastFoundationStereo)
│   ├── extractor.py           # 特征提取器 (EdgeNeXt)
│   ├── geometry.py            # 几何编码体积
│   ├── update.py              # GRU 更新模块
│   ├── submodule.py           # 基础模块 (卷积、注意力等)
│   └── utils/
│       ├── frame_utils.py    # 帧处理工具
│       └── utils.py          # 通用工具函数
├── scripts/
│   ├── run_demo.py           # Demo 运行脚本
│   └── make_onnx.py          # ONNX 导出脚本
├── weights/                   # 模型权重目录
├── doc/
│   ├── Fast-FoundationStereo_2026.pdf  # 论文
│   └── FoundationStereo_2025.pdf       # 前作论文
├── svo_to_ply.py              # SVO→PLY 主程序 (本项目)
├── filter_ply.py              # PLY 滤波工具 (本项目)
└── readme.md                 # 项目说明
```

---

## 总结

Fast-FoundationStereo 是当前最先进的实时零样本立体匹配模型之一，其核心优势在于：

1. **零样本泛化**: 无需针对特定场景微调即可工作
2. **实时性能**: TensorRT 加速后可达 60+ FPS
3. **层级推理**: 粗到精策略平衡速度与精度
4. **轻量级**: 仅 14.6M 参数，适合边缘部署

本项目将其与 ZED SVO 文件深度集成，实现了从 SVO 到高质量 PLY 点云的完整重建管线 ✧(≖ ◡ ≖✿)
