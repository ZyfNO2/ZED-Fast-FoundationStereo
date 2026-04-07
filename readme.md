# ZED SVO → PLY 点云重建工具

基于 [Fast-FoundationStereo](https://github.com/NVLabs/Fast-FoundationStereo) 的 ZED 相机 SVO 文件转 PLY 点云重建工具。

## 效果预览

![重建效果2](重建2.png)

## 功能特性

- **SVO 文件读取**：直接从 ZED SVO 文件提取左右目图像、相机内参、外参（位姿矩阵）
- **深度估计**：使用 Fast-FoundationStereo 零样本立体匹配模型推理深度图
- **坐标转换**：自动处理 OpenCV 坐标系与 ZED 坐标系的转换
- **多级滤波**：
  - 深度边缘滤波（去除物体边缘不可靠点）
  - 稀疏空间滤波（去除离群点）
  - 统计滤波（Statistical Outlier Removal）
  - 半径滤波（Radius Outlier Removal）
  - DBSCAN 聚类滤波
  - 锥形放射检测与去除
  - 时序双向一致性滤波
- **中间结果保存**：每步滤波后自动保存中间PLY文件，便于对比效果

## 快速开始

### 安装依赖

```bash
pip install torch torchvision
pip install timm einops omegaconf scipy numpy scikit-image opencv-contrib-python imageio pyyaml open3d
pip install pyzed
```

### 推荐参数运行

```bash
python svo_to_ply.py --svo "path/to/your.svo2" \
    --depth_edge_threshold 0.01 \
    --temporal_warmup_frames 0 --temporal_min_half_frames 1 \
    --nb_neighbors 100 --std_ratio 0.2 \
    --minimal_filtering
```

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--svo` | SVO 文件路径 | 必需 |
| `--scale` | 图像缩放比例 | 0.5 |
| `--frame_skip` | 帧采样间隔 | 5 |
| `--depth_edge_threshold` | 深度边缘阈值 (0.01=1cm) | 0.1 |
| `--temporal_warmup_frames` | 时序滤波预热帧数 | 5 |
| `--temporal_min_half_frames` | 前后各需出现的最小帧数 | 2 |
| `--minimal_filtering` | 跳过体素下采样，只保留统计/半径/DBSCAN滤波 | False |
| `--nb_neighbors` | 统计滤波邻域点数 | 100 |
| `--std_ratio` | 统计滤波标准差阈值 | 0.2 |

### 迭代优化流程

1. 运行 `svo_to_ply.py` 生成 `_02_temporal.ply`
2. 使用 `filter_ply.py` 尝试不同滤波参数
3. 在 CloudCompare 中对比效果
4. 确定最佳参数后更新默认配置

```bash
# 示例：迭代优化
python filter_ply.py "output/your_svo_02_temporal.ply" --stat_std 0.2
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `svo_to_ply.py` | 主程序：SVO 转 PLY |
| `filter_ply.py` | 辅助工具：对已有PLY文件施加不同滤波参数 |
| `output/` | 输出目录，包含中间PLY文件 |

## 中间滤波结果

运行后会生成带编号的中间结果文件：

```
output/
├── your_svo_01_sparse.ply          # 稀疏滤波后
├── your_svo_02_temporal.ply         # 时序双向滤波后
├── your_svo_03_statistical.ply      # 统计滤波后
├── your_svo_04_radius.ply           # 半径滤波后
├── your_svo_05_dbscan.ply           # DBSCAN聚类后
└── your_svo.ply                     # 最终结果
```

## 技术原理

1. **深度估计**：使用 Fast-FoundationStereo 零样本立体匹配模型，从左右目图像对估计深度图
2. **坐标转换**：`depth2xyzmap` 产生的 OpenCV 坐标系 (X→右, Y↓, Z→前) 需转换为 ZED 的 RIGHT_HANDED_Y_UP 坐标系
3. **点云融合**：使用 ZED 位姿矩阵将每帧点云从相机坐标系转换到世界坐标系
4. **滤波策略**：先时序过滤（去除单视角噪声），再空间滤波（保留主要结构）

## 参考

- [Fast-FoundationStereo](https://github.com/NVLabs/Fast-FoundationStereo) - CVPR 2026
- [ZED SDK](https://www.stereolabs.com/developers) - Stereolabs

## 致谢

感谢黑塔大人（Herta）的天才指导，使得本项目能够顺利完成。在黑塔大人的引领下，即使是开拓者也能触及星辰的高度 ✧(≖ ◡ ≖✿)
