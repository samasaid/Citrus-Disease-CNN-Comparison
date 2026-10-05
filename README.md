# Citrus Disease CNN Comparison

This repository contains the source code used in the study:

**“A Comparative Study of Lightweight and Deep CNN Architectures for Robust Citrus Disease Classification.”**

The code was applied to both **citrus fruit and citrus leaf image datasets** using the same unified experimental protocol. It supports the main computational procedures reported in the manuscript, including data preparation, CNN model training, performance evaluation, five-fold cross-validation, robustness analysis, statistical comparisons, and Grad-CAM-based interpretability analysis.

## Code Organization

```text
code/
├── 01_data_preparation.py
├── 02_training.py
├── 03_evaluation.py
├── 04_cross_validation.py
├── 05_robustness.py
└── 06_gradcam.py
```

### 01_data_preparation.py

Data preprocessing, dataset organization, duplicate/near-duplicate checking, and train/validation/test splitting for the citrus fruit and leaf datasets.

### 02_training.py

Training and fine-tuning of the four evaluated CNN architectures:

* EfficientNetB0
* MobileNetV2-0.35
* ResNet50
* DenseNet121

The same training protocol was applied to both the fruit and leaf datasets.

### 03_evaluation.py

Calculation of the classification performance measures reported in the manuscript, including accuracy, precision, recall, F1-score, Macro-F1, multiclass ROC-AUC, expected calibration error (ECE), and bootstrap confidence intervals.

### 04_cross_validation.py

Five-fold cross-validation used to assess the consistency of model performance across the development partitions. The code also performs statistical comparisons between model performances, including the Wilcoxon signed-rank test with Holm correction, as reported in the manuscript.

### 05_robustness.py

Evaluation of model robustness under the controlled image perturbations described in the manuscript for the selected architectures.

### 06_gradcam.py

Grad-CAM-based qualitative analysis of model activation patterns for the selected architectures.

## Experimental Setup

The experiments were conducted on both citrus fruit and citrus leaf image datasets using a unified experimental protocol.

Images were resized to **224 × 224 pixels**, and a fixed random seed of **42** was used. The same training and evaluation procedures were applied across the four evaluated CNN architectures, with architecture-specific preprocessing for ImageNet-pretrained models.

The training procedure used transfer learning followed by fine-tuning.

## Datasets

The datasets used in the study are not redistributed in this repository. The original sources and dataset preparation procedures are described in the manuscript and implemented in the data preparation code.

## Reproducibility

The software dependencies required to run the code are listed in `requirements.txt`.

The code is provided to support reproducibility of the computational experiments reported in the manuscript across both citrus fruit and citrus leaf datasets.
