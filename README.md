# DeepFM & WideDeep for CTR Prediction

基于 Criteo DAC 数据集的点击率（CTR）预测项目，支持 DeepFM 和 WideDeep 两种模型，支持单卡/多卡 DDP 分布式训练。

CTR prediction project on the Criteo DAC dataset, featuring DeepFM and WideDeep models with single-GPU and multi-GPU DDP distributed training support.

---

## 项目简介 / Overview

本项目使用 Criteo Display Advertising Challenge (DAC) 数据集，实现了两种经典的 CTR 预测模型：

- **DeepFM** — 融合 FM（Factorization Machine）的二阶特征交叉与深度神经网络（DNN），同时捕捉低阶和高阶特征交互
- **WideDeep** — Wide 部分通过类别特征两两交叉编码学习显式特征组合，Deep 部分通过 Embedding + MLP 学习隐式高阶特征

This project implements two classic CTR prediction models on the Criteo DAC dataset:

- **DeepFM** — Combines FM's second-order feature interactions with a deep neural network (DNN) to capture both low-order and high-order feature interactions
- **WideDeep** — The Wide component learns explicit feature crosses via pairwise category feature interactions, while the Deep component learns implicit high-order features through Embedding + MLP

---

## 数据集 / Dataset

**Criteo DAC** — Criteo 公司 7 天的展示广告流量数据，约 4500 万行。

Each row: `<label> <I1> ... <I13> <C1> ... <C26>` (tab-separated)

| 类型 / Type | 数量 / Count | 说明 / Description |
|---|---|---|
| Label | 1 | 是否点击 (0/1) / Clicked or not |
| Integer Features (I1-I13) | 13 | 整数型特征（多数为计数类）/ Integer features (mostly counts) |
| Categorical Features (C1-C26) | 26 | 类别特征（哈希到 32 位）/ Categorical features (hashed to 32 bits) |

**下载地址 / Download**: [Kaggle Criteo DAC](https://www.kaggle.com/c/criteo-display-ad-challenge/)

将 `train.txt` 放入 `data/dac/` 目录下。Place `train.txt` in the `data/dac/` directory.

---

## 项目结构 / Project Structure

```
├── main.py              # DeepFM 训练入口 / DeepFM training entry
├── main_widedeep.py     # WideDeep 训练入口 / WideDeep training entry
├── net.py               # 模型定义 (MLP, FM, DeepFM, WideDeep) / Model definitions
├── readdata.py          # 数据读取与预处理 / Data loading and preprocessing
├── test.py              # 工具函数 (提取 AUC) / Utility functions
└── data/dac/
    └── train.txt        # Criteo DAC 数据集 / Criteo DAC dataset
```

---

## 环境依赖 / Dependencies

```bash
Python >= 3.9
PyTorch >= 2.0
pandas
numpy
scikit-learn
```

---

## 使用方法 / Usage

### 单卡训练 / Single GPU Training

自动选择显存最空闲的 GPU。Automatically selects the GPU with the most free memory.

```bash
# DeepFM
python main.py

# WideDeep
python main_widedeep.py
```

### 多卡 DDP 训练 / Multi-GPU DDP Training

```bash
# DeepFM, N = GPU 数量
torchrun --nproc_per_node=N main.py

# WideDeep
torchrun --nproc_per_node=N main_widedeep.py

# 指定空闲 GPU / Specify GPUs
CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 main_widedeep.py
```

程序启动时会自动扫描 GPU 显存占用，并推荐可用的 GPU。The program scans GPU memory usage at startup and recommends available GPUs.

---

## 模型架构 / Model Architecture

### DeepFM

```
Input (39 features)
├── FM Layer (一阶线性 + 二阶交叉) / FM Layer (1st-order linear + 2nd-order interaction)
└── Deep Layer (Embedding → MLP [512, 256, 128, 64, 32] → 1)
Output = FM + Deep → Sigmoid
```

### WideDeep

```
Input (39 features)
├── Wide: 类别特征两两交叉 → Embedding(4) × 325 → Linear / Pairwise category crosses → Embedding(4) × 325 → Linear
└── Deep: 类别+数值 Embedding → MLP [512, 256, 128, 64, 32] → 1 / Category+numeric Embedding → MLP → 1
Output = Wide + Deep → Sigmoid
```

---

## 数据预处理 / Data Preprocessing

1. **数值特征** / Numeric features: `log1p` 变换 + Z-Score 标准化（基于训练集统计量）
2. **类别特征** / Categorical features: 基于训练集词频（min_freq=5）编码为整数索引，缺失值编码为 0
3. **x_val**: 类别特征 one-hot 标记（有值=1，缺失=0）+ 数值特征，拼接为 39 维输入
4. 数据集按 90:10 划分为训练集和验证集（按时间顺序）

---

## 训练配置 / Training Configuration

| 参数 / Parameter | 值 / Value |
|---|---|
| Embedding Dimension | 32 |
| Batch Size | 2048 (per GPU) |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |
| Epochs | 200 |
| Eval Every | 5 epochs |
| Early Stop Patience | 10 evals (50 epochs) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Gradient Clipping | max_norm=5.0 |
| Loss Function | BCEWithLogitsLoss |
| Evaluation Metric | AUC |

---

## 输出文件 / Output Files

| 文件 / File | 说明 / Description |
|---|---|
| `best_deepfm.pth` / `best_widedeep.pth` | 最佳模型参数 / Best model checkpoint |
| `best_auc.txt` / `best_auc_widedeep.txt` | 最佳 AUC 记录 / Best AUC record |
| `loss.txt` / `loss_widedeep.txt` | 每 epoch 训练 loss / Per-epoch training loss |
| `auc.txt` / `auc_widedeep.txt` | 每次验证的 AUC 记录 / Validation AUC records |

---

## 实验结果 / Experiment Results

在 Criteo DAC 数据集上，90% 训练 / 10% 验证，按时间顺序划分。

On the Criteo DAC dataset, 90% train / 10% validation, split chronologically.

### 结果汇总 / Summary

| 模型 / Model | Best AUC | 达到 Epoch / Epoch Reached | Early Stop |
|---|---|---|---|
| **DeepFM** | 0.8039 | 50 | Epoch 100 |
| **WideDeep** | **0.8056** | 45 | - |

WideDeep 略优于 DeepFM（+0.0017 AUC）。

WideDeep slightly outperforms DeepFM (+0.0017 AUC).

### DeepFM AUC 曲线 / DeepFM AUC Curve

| Epoch | AUC | Best |
|---|---|---|
| 5 | 0.7993 | 0.7993 |
| 10 | 0.8020 | 0.8020 |
| 15 | 0.8027 | 0.8027 |
| 20 | 0.8029 | 0.8029 |
| 25 | 0.8029 | 0.8029 |
| 30 | 0.8029 | 0.8029 |
| 35 | 0.8029 | 0.8029 |
| 40 | 0.8029 | 0.8029 |
| 45 | 0.8037 | 0.8037 |
| 50 | 0.8039 | 0.8039 |
| 55 | 0.8035 | 0.8039 |
| 60 | 0.8038 | 0.8039 |
| 65 | 0.8032 | 0.8039 |
| 70 | 0.8034 | 0.8039 |
| 75 | 0.7942 | 0.8039 |
| 80 | 0.7843 | 0.8039 |
| 85 | 0.7802 | 0.8039 |
| 90 | 0.7671 | 0.8039 |
| 95 | 0.6939 | 0.8039 |
| 100 | 0.6684 | 0.8039 |

### WideDeep AUC 曲线 / WideDeep AUC Curve

| Epoch | AUC | Best |
|---|---|---|
| 5 | 0.8039 | 0.8039 |
| 10 | 0.8040 | 0.8040 |
| 15 | 0.8046 | 0.8046 |
| 20 | 0.8047 | 0.8047 |
| 25 | 0.8045 | 0.8047 |
| 30 | 0.8045 | 0.8047 |
| 35 | 0.8044 | 0.8047 |
| 40 | 0.8053 | 0.8053 |
| 45 | 0.8056 | 0.8056 |
| 50 | 0.8053 | 0.8056 |

---

## License

MIT
