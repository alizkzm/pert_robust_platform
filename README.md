
# REPT: A Robust Evaluation Platform for Transferability


---

## Introduction

Transferability scores aim to quantify how well a model trained on one domain generalizes to a target domain. Despite numerous methods proposed for measuring transferability, their reliability and practical usefulness remain inconclusive, often due to differing experimental setups, datasets, and assumptions. In this paper, we introduce REPT (Robust Evaluation Platform of Transferability), a comprehensive benchmarking framework designed to systematically evaluate transferability scores across diverse settings.

---

## Features

- Evaluation of multiple transferability metrics.
- Support for specifying the source domain of pre-trained models.
- Comparative evaluation of head-only vs full fine-tuning.
- Control over the complexity of the model hub based on model parameters.
- Easy-to-use scripts for feature extraction, metric evaluation, and transferability estimation.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchvision 0.15+
- numpy 1.24+
- pandas 1.5+
- scikit-learn 1.2+
- timm 0.9+
- tqdm 4.65+

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## Argument Definitions

| Argument | Meaning |
|----------|---------|
| `-s`     | Specifies the source domain of the pre-trained model (e.g., `imagenet`). |
| `-f`     | Specifies the ground-truth type used for transferability estimation. Options: `head-training`, `full-training`, etc. |
| `-c`     | Specifies the model-hub complexity degree based on model parameter count: <br> `1` → 7 models with the lowest parameters <br> `5` → 7 models with the highest parameters <br> `2-4` → Intermediate selections from low to high complexity. |
| `-d`     | Specifies the target dataset name (e.g., `caltech101`, `cifar10`, etc.). |
| `-m`     | Specifies the source model name (e.g., `resnet18`, `densenet121`, etc.). |
| `-me`     | Specifies the transferability score (e.g., `sfda`, `wasserstein`, etc.). |

---

## Usage Examples

### 1. Extract Features

```bash
python forward_feature_group1.py -s imagenet -m resnet34 -d cifar100
```

### 2. Evaluate Metric

```bash
python evaluate_metric_group1_cpu.py -s imagenet -me sfda -d cifar100
```

### 3. Estimate Transferability

```bash
python tw_group1_cpu.py -s imagenet -f head-training -c 1 -d cifar100
```

---

