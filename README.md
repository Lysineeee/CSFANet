# CSFANet
Learning Cross‑Scale Spatial Feature Aggregation for Remote Sensing Change Detection

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://link.springer.com/chapter/10.1007/978-981-92-3444-8_43)
[![Dataset-LEVIR‑CD](https://img.shields.io/badge/Dataset-LEVIR--CD-green)](https://justchenhao.github.io/LEVIR/)
[![Dataset-GZ‑CD](https://img.shields.io/badge/Dataset-GZ--CD-orange)](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)

> Remote Sensing Change Detection (RSCD) aims to identify surface changes by comparing bi‑temporal remote sensing images. Although convolutional neural networks (CNNs) have significantly improved detection performance, their limited receptive fields and inherently sequential architectures hinder effective cross‑layer interaction, making multi‑scale feature fusion challenging. To address this issue, this paper propose a novel **Cross‑scale Spatial Feature Aggregation Network (CSFANet)**. Built upon a ResNet‑18 backbone, bi‑temporal features are first extracted, followed by a **Cross‑scale Spatial Feature Aggregation Module (CSFAM)** that effectively integrates low‑level spatial details with high‑level semantic information across different layers.

## Abstract
Remote Sensing Change Detection (RSCD) aims to identify surface changes by comparing bi‑temporal remote sensing images. Although convolutional neural networks (CNNs) have significantly improved detection performance, their limited receptive fields and inherently sequential architectures hinder effective cross‑layer interaction, making multi‑scale feature fusion challenging. To address this issue, this paper propose a novel Cross‑scale Spatial Feature Aggregation Network, termed **CSFANet**. Built upon a ResNet‑18 backbone, bi‑temporal features are first extracted, followed by a Cross‑scale Spatial Feature Aggregation Module (CSFAM) that effectively integrates low‑level spatial details with high‑level semantic information across different layers.

Experimental results on the LEVIR‑CD and GZ‑CD datasets demonstrate that CSFANet consistently outperforms existing methods across multiple evaluation metrics, achieving F1‑scores of **90.91%** and **88.94%**, respectively, surpassing the ISDANet baseline.

**Keywords**: Remote sensing change detection · Multi‑scale feature fusion · Bi‑temporal learning

---

## 1. Introduction
Remote sensing change detection (RSCD) aims to identify surface changes, such as vegetation variation, infrastructure development, and building expansion, by comparing bi‑temporal or multi‑temporal images acquired over the same geographic area at different times. It plays a pivotal role in applications including post‑disaster assessment, land use and land cover monitoring and management, urban expansion analysis, and environmental change tracking.

Challenges in RSCD:
- Illumination, seasonal variations, atmospheric effects, geometric distortions, viewpoint differences, pseudo‑changes (shadows, misalignments, spectral shifts).
- Low‑level features: preserve fine spatial details and boundaries, but sensitive to noise and pseudo‑changes.
- High‑level features: rich semantic information, robust to disturbance, lack precise spatial localization, hard for small change targets.

Existing multi‑scale fusion limitations:
- SNUNet‑CD: fuse features only at corresponding scales, no full cross‑level aggregation.
- SEIFNet, ISDANet: only fuse adjacent levels or local attention, insufficient multi‑level context.
- Transformer‑based methods (ChangeFormer, BIT): good global modeling, lose fine‑grained details during decoding.

### Main Contributions
1. Propose a weight‑shared **Cross‑scale Spatial Feature Aggregation Module (CSFAM)** that aligns multi‑level features into a unified spatial domain for full‑resolution aggregation.
2. Introduce a **position‑aware feature enhancement strategy** to explicitly encode spatial information across feature scales.
3. Extensive experiments on two public datasets (LEVIR‑CD, GZ‑CD) demonstrate that CSFANet outperforms state‑of‑the‑art approaches.

---

## 2. Method

### 2.1 Overall Architecture
![CSFANet Overview](./figures/fig1_arch.png)
> Figure 1: Overview of the proposed CSFANet architecture.

The backbone is ResNet‑18. For bi‑temporal input $T_1$, the output features from four backbone stages are:
$$
X_{i} \in \mathbb {R}^{C_{i} \times H_{i} \times W_{i}},\ i=1,2,3,4 \tag{1}
$$
where $X_i$ denotes feature map from $i$-th stage; $C_i$, $H_i$, $W_i$ denote channel number, height, width.

> We take ISDANet as our baseline, all improvements are built upon this baseline.

### 2.2 Attention‑Guided Feature Enhancement and Spatial Alignment

1. **Position Embedding**: inject spatial location information
$$
\tilde{X}_{i}=X_{i}+PE\left(X_{i}\right),\ i=1,2,3,4 \tag{2}
$$

2. **CBAM attention recalibration**: adaptively recalibrate channel‑wise and spatial responses
$$
X_{i}^{att }=CBAM\left(\tilde{X}_{i}\right),\ i=1,2,3,4 \tag{3}
$$

3. **Spatial alignment**: map all features to target scale $i$
$$
\hat{X}_{j}^{i}=\mathcal{R}_{i}\left(X_{j}^{att }\right),\ i, j \in\{1,2,3,4\} \tag{4}
$$
- High‑resolution features: downsample via cropping‑and‑concatenation along channel dimension.
- Low‑resolution features: upsample via bilinear interpolation.

### 2.3 Multi‑Scale Feature Aggregation

1. Concatenate all aligned features at target scale $i$ along channel dimension:
$$
X_{cat }^{i}=Cat\left(\hat{X}_{1}^{i},\ \hat{X}_{2}^{i},\ \hat{X}_{3}^{i},\ \hat{X}_{4}^{i}\right),\ i=1,2,3,4 \tag{5}
$$

2. Convolution fusion block ($1\times1$ for channel compression, $3\times3$ for spatial fusion):
$$
X_{fused }^{i}=Conv_{3 \times 3}\left(Conv_{1 \times 1}\left(X_{cat}^{i}\right)\right) \tag{6}
$$

Full CSFAM formulation:
$$
\left\{X_{fused }^{i}\right\}_{i=1}^{4}=\mathcal{F}\left(\left\{X_{i}\right\}_{i=1}^{4}\right) \tag{7}
$$

> The same weight‑shared CSFAM module is applied to $T_2$ branch for symmetric bi‑temporal feature fusion.

---

## 3. Experiments

### 3.1 Datasets
1. **LEVIR‑CD**
- 637 bi‑temporal image pairs, resolution $0.5\,\text{m/pixel}$
- Original size $1024\times1024$, cropped to $256\times256$ patches
- Split ratio: $7:1:2$ → train:7120 / val:1024 / test:2048

2. **GZ‑CD**
- 19 large bi‑temporal remote sensing images of Guangzhou suburbs, resolution ~$0.55\,\text{m/pixel}$
- Cropped to $256\times256$ patches
- Split ratio: $8:1:1$ → train:2504 / val:313 / test:313 (small‑sample scenario)

### 3.2 Experimental Settings
- OS: Ubuntu, GPU: RTX‑3090 (24GB)
- Augmentation: random crop, rotation, flip
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam
- Epochs: 200, batch size: 8
- Initial learning rate: $1\times10^{-4}$
- Polynomial decay:
$$
lr = lr_0 \times\left(1-\frac{epoch}{max\_epoch}\right)^{power},\quad power=0.9
$$
- Evaluation metrics: Precision(PR), Kappa, IoU, F1‑score(F1), FNR, ER.

### 3.3 Quantitative Results

#### LEVIR‑CD
| Methods | PR(%) | Kappa(%) | IoU(%) | F1(%) | FNR(%) | ER(%) |
|--------|-------|----------|--------|-------|--------|-------|
| FC‑EF | 85.58 | 82.30 | 71.19 | 83.17 | 19.11 | 1.67 |
| FC‑Siam‑Diff | 89.49 | 84.08 | 73.69 | 84.85 | 19.33 | 1.47 |
| FC‑Siam‑Conc | 86.76 | 85.56 | 75.89 | 86.29 | 14.17 | 1.39 |
| ChangeNet | 91.63 | 88.63 | 80.49 | 89.19 | 13.12 | 1.07 |
| DSIFN | 91.53 | 87.75 | 79.12 | 88.34 | 14.30 | 1.13 |
| BIT | 91.26 | 89.33 | 81.59 | 89.86 | 11.49 | 1.02 |
| SNUNet | 91.51 | 89.46 | 81.79 | 89.98 | 11.49 | 1.00 |
| ICIFNet | 91.31 | 89.16 | 81.24 | 89.18 | 12.77 | 1.44 |
| DMINet | 92.02 | 89.31 | 81.56 | 89.85 | 12.27 | 1.21 |
| SAGNet | 91.79 | 89.58 | 81.98 | 90.10 | 11.24 | 0.98 |
| SFBI‑Net | 91.85 | 89.45 | 83.05 | 90.65 | 10.54 | 1.08 |
| SEIFNet | 92.49 | 89.29 | 83.40 | 90.25 | 10.65 | 1.79 |
| ISDANet | 91.64 | 89.59 | 82.01 | 90.12 | 11.36 | 0.99 |
| **Ours(CSFANet)** | **91.46** | **90.11** | **83.84** | **90.91** | **10.22** | **0.95** |

> CSFANet gains +0.26% F1, +0.44% IoU compared with competitors on LEVIR‑CD.

Visual comparison on LEVIR‑CD:
![LEVIR‑CD Visual Result](./figures/fig2_levir_vis.png)
> Figure 2: Visual comparisons on LEVIR‑CD dataset. White: true positives; Red: false positives; Black: true negatives; Green: false negatives.

#### GZ‑CD
| Methods | PR(%) | Kappa(%) | IoU(%) | F1(%) | FNR(%) | ER(%) |
|--------|-------|----------|--------|-------|--------|-------|
| FC‑EF | 79.86 | 69.44 | 56.24 | 71.99 | 34.47 | 4.72 |
| FC‑Siam‑Diff | 82.70 | 65.55 | 51.72 | 68.18 | 42.01 | 5.01 |
| FC‑Siam‑Conc | 82.16 | 68.67 | 55.26 | 71.19 | 37.20 | 4.71 |
| ChangeNet | 88.63 | 84.32 | 75.01 | 85.72 | 17.01 | 2.56 |
| DSIFN | 89.48 | 79.83 | 68.76 | 81.49 | 24.54 | 3.09 |
| BIT | 86.80 | 82.80 | 72.94 | 84.35 | 17.96 | 2.82 |
| SNUNet | 89.00 | 85.54 | 76.75 | 86.85 | 15.20 | 2.38 |
| ICIFNet | 88.09 | 83.05 | 73.25 | 84.56 | 18.69 | 2.75 |
| DMINet | 86.62 | 83.17 | 73.45 | 84.70 | 17.15 | 2.77 |
| SAGNet | 89.56 | 84.98 | 75.91 | 86.30 | 15.95 | 2.42 |
| SFBI‑Net | 89.03 | 85.35 | 76.05 | 86.36 | 15.38 | 2.62 |
| SEIFNet | 88.98 | 85.62 | 76.23 | 86.65 | 15.12 | 2.57 |
| ISDANet | 89.72 | 87.07 | 78.97 | 88.25 | 13.17 | 2.14 |
| **Ours(CSFANet)** | **90.97** | **87.93** | **80.08** | **88.94** | **13.01** | **1.85** |

> CSFANet gains +0.69% F1, +1.11% IoU compared with competitors on GZ‑CD (small‑sample condition).

Visual comparison on GZ‑CD:
![GZ‑CD Visual Result](./figures/fig3_gzcd_vis.png)
> Figure3: Visual comparisons on GZ‑CD dataset. White: true positives; Red: false positives; Black: true negatives; Green: false negatives.

### 3.4 Ablation Study
| Dataset | Methods | CSFAM | Spatial Alignment | Position Embedding | IoU | F1 |
|---------|---------|-------|-------------------|-------------------|-----|----|
| LEVIR‑CD | Baseline+ |  |  |  | 82.01 | 90.12 |
| LEVIR‑CD | Baseline+ | ✓ |  |  | 82.47 | 90.23 |
| LEVIR‑CD | Baseline+ | ✓ | ✓ |  | 82.55 | 90.36 |
| LEVIR‑CD | Baseline+ | ✓ | ✓ | ✓ | **83.84** | **90.91** |
| GZ‑CD | Baseline+ |  |  |  | 78.97 | 88.25 |
| GZ‑CD | Baseline+ | ✓ |  |  | 79.19 | 88.39 |
| GZ‑CD | Baseline+ | ✓ | ✓ |  | 79.62 | 88.76 |
| GZ‑CD | Baseline+ | ✓ | ✓ | ✓ | **80.08** | **88.94** |

Observations:
1. CSFAM brings consistent performance gain on both datasets.
2. Spatial alignment and position embedding are complementary, removing either leads to obvious drop.
3. Position embedding encodes spatial location, alleviates misalignment‑caused errors.

---

## 4. Conclusion
We propose CSFANet with Cross‑scale Spatial Feature Aggregation Module (CSFAM) for remote sensing change detection.
- CSFAM aggregates multi‑scale features in unified spatial domain, enabling interaction between low‑level spatial details and high‑level semantic representations.
- Spatial alignment + position embedding maintain spatial consistency and reduce information loss during fusion.
- Experiments on LEVIR‑CD and GZ‑CD show superior performance; our method achieves better boundary accuracy and pseudo‑change suppression.

> Although numerical improvements are moderate, the model significantly improves boundary delineation and suppress pseudo‑changes.

---

## 5. Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{csfanet,
  title={Learning Cross‑Scale Spatial Feature Aggregation for Remote Sensing Change Detection},
  author={Anonymous},
  booktitle={},
  year={2026}
}
```
