# CSFANet: Learning Cross-Scale Spatial Feature Aggregation for Remote Sensing Change Detection

## Overview

CSFANet is a deep learning framework for **Remote Sensing Change Detection (RSCD)** that addresses the limitation of conventional CNN-based methods in integrating low-level spatial details and high-level semantic representations.

The model introduces a **Cross-Scale Feature Aggregation Module (CSFAM)** to perform feature interaction in a unified spatial domain, improving detection performance for:

- Small-scale changes
    
- Subtle structural changes
    
- Boundary delineation
    
- Pseudo-change suppression
    

---

## Motivation

Remote sensing change detection aims to identify surface changes between bi-temporal images.

Typical CNN-based approaches suffer from:

- Limited receptive fields
    
- Weak interaction between hierarchical features
    
- Loss of fine spatial structures
    
- Difficulty preserving subtle changes
    

Lower-level features:

- Rich spatial details
    
- Precise localization
    
- Sensitive to noise and pseudo-changes
    

Higher-level features:

- Rich semantic information
    
- Strong robustness
    
- Weak boundary precision
    

CSFANet is designed to bridge this gap through effective cross-scale feature aggregation.

---

## Architecture

### Overall Framework

The network adopts a Siamese architecture based on **ResNet-18** backbone:

```
Input(T1,T2)
        ↓
ResNet-18 Encoder
        ↓
Multi-level features:
{X1,X2,X3,X4}
        ↓
CSFAM
    ├── Position Embedding
    ├── CBAM Enhancement
    ├── Spatial Alignment
    ├── Cross-scale Aggregation
        ↓
Decoder
        ↓
Change Map
```

---

## Cross-Scale Feature Aggregation Module (CSFAM)

The proposed module contains several stages:

### 1. Position-aware Feature Enhancement

Position embedding is introduced to inject spatial information:

$$
X_i' = X_i + PE(X_i)  
$$

Benefits:

- Spatial awareness
    
- Improved boundary sensitivity
    
- Better feature correspondence
    

---

### 2. Attention-guided Feature Enhancement

CBAM is applied:


$$X_i^{att}=CBAM(X_i')  $$


Benefits:

- Channel attention
    
- Spatial attention
    
- Noise suppression
    

---

### 3. Spatial Alignment

Features from different scales are aligned into a common spatial domain:

- High-resolution features:
    
    - Cropping + concatenation
        
- Low-resolution features:
    
    - Bilinear interpolation
        

Benefits:

- Reduced information loss
    
- Full cross-scale interaction
    

---

### 4. Cross-scale Feature Fusion

Aligned features are concatenated:


$$X_i^{cat}=Cat(X_1,X_2,X_3,X_4)  $$


Fusion block:

```
1×1 Conv
      ↓
3×3 Conv
      ↓
BN + ReLU
```

Output:


$$X_i^{fused}  $$


---

## Main Contributions

1. Introduces a weight-sharing Cross-Scale Feature Aggregation Module (CSFAM)
    
2. Unifies multi-scale features into a common spatial domain
    
3. Introduces position-aware feature enhancement
    
4. Achieves superior performance on public RSCD datasets
    

---

## Dataset

### LEVIR-CD

- Resolution: 0.5 m/pixel
    
- Training: 7120 image pairs
    
- Validation: 1024 image pairs
    
- Testing: 2048 image pairs
    

### GZ-CD

- Resolution: 0.55 m/pixel
    
- Training: 2504 image pairs
    
- Validation: 313 image pairs
    
- Testing: 313 image pairs
    

---

## Training Configuration

|Parameter|Value|
|---|---|
|Backbone|ResNet-18|
|Optimizer|Adam|
|Loss|BCEWithLogitsLoss|
|Batch Size|8|
|Epochs|200|
|Initial LR|1e-4|
|LR Scheduler|Polynomial|
|Power|0.9|

Data augmentation:

- Random crop
    
- Rotation
    
- Horizontal flip
    
- Vertical flip
    

---

## Evaluation Metrics

The following metrics are used:

- Precision (PR)
    
- False Negative Rate (FNR)
    
- F1-score
    
- Intersection over Union (IoU)
    
- Error Rate (ER)
    
- Kappa coefficient
    

---

## Experimental Results

### LEVIR-CD

|Method|IoU (%)|F1 (%)|
|---|---|---|
|ISDANet|82.01|90.12|
|SEIFNet|83.40|90.25|
|SFBI-Net|83.05|90.65|
|CSFANet (Ours)|**83.84**|**90.91**|

Improvement over strongest baseline:

- +0.26% F1
    
- +0.44% IoU
    

---

### GZ-CD

|Method|IoU (%)|F1 (%)|
|---|---|---|
|ISDANet|78.97|88.25|
|SNUNet|76.75|86.85|
|SEIFNet|76.23|86.65|
|CSFANet (Ours)|**80.08**|**88.94**|

Improvement over strongest baseline:

- +0.69% F1
    
- +1.11% IoU
    

---

## Ablation Study

### LEVIR-CD

|Method|IoU|F1|
|---|---|---|
|Baseline|82.01|90.12|
|+ CSFAM|82.47|90.23|
|+ CSFAM + Spatial Alignment|82.55|90.36|
|+ Full Model|**83.84**|**90.91**|

### GZ-CD

|Method|IoU|F1|
|---|---|---|
|Baseline|78.97|88.25|
|+ CSFAM|79.19|88.39|
|+ CSFAM + Spatial Alignment|79.62|88.76|
|+ Full Model|**80.08**|**88.94**|

---

## Key Findings

- Cross-scale aggregation improves multi-level feature interaction
    
- Position embedding strengthens spatial consistency
    
- Better preservation of fine structures
    
- Reduced pseudo-changes
    
- Strong generalization under small-sample scenarios
    

---

## Code

Official implementation:

[https://github.com/Lysineeee/CSFANet](https://github.com/Lysineeee/CSFANet)

---

## Acknowledgement

This work is supported by:

- Guangxi Graduate Education Innovation Project (YCBZ2025156)
    
- Guangxi Natural Science Foundation (2025GXNSFAA069491)
    
- National Natural Science Foundation of China (62361021)
