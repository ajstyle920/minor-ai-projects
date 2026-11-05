# 🖼️ CNN CIFAR-10 Image Classification

A **Convolutional Neural Network built from scratch in PyTorch** that classifies 32×32 colour images into 10 object categories from the **CIFAR-10** dataset. It reaches **74.8% test accuracy** after 10 epochs.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Image%20Classification-blue)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-74.76%25-success)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Results at a Glance](#-results-at-a-glance)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Preprocessing](#-preprocessing)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Limitations & Future Work](#-limitations--future-work)
- [Author](#-author)

---

## 🔍 Overview

This project is a hands-on introduction to **computer vision with CNNs**. A three-block convolutional network learns visual features (edges and textures at first, then shapes and object parts) directly from raw pixels, and a fully connected classifier maps those features to one of ten classes.

It covers the complete PyTorch vision workflow: loading datasets with `torchvision`, image transforms, `DataLoader` batching, a custom `nn.Module` with convolutional and fully connected blocks, a training loop written by hand, and accuracy evaluation.

---

## 🏆 Results at a Glance

| Metric | Value |
|--------|:-----:|
| **Test Accuracy** | **74.76%** (7,476 / 10,000) |
| Random-guess baseline | 10% |
| Final Training Loss | 0.147 |
| Trainable Parameters | ~620K |
| Epochs | 10 |

---

## 🎯 Problem Statement

**Given** a 32 × 32 RGB image, **predict** which of 10 classes it belongs to:

| | | | | |
|:-:|:-:|:-:|:-:|:-:|
| ✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer |
| 🐶 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚚 Truck |

This is a **multi-class image classification** problem.

---

## 📊 Dataset

**[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)**, loaded with `torchvision.datasets.CIFAR10`:

| Property | Value |
|----------|-------|
| Total images | 60,000 |
| Training set | 50,000 |
| Test set | 10,000 |
| Classes | 10 (6,000 images each, balanced) |
| Image size | 32 × 32 × 3 (RGB) |

> 📥 The dataset is **downloaded automatically** to `./data/` the first time the notebook runs. No manual setup is needed.

---

## 🧹 Preprocessing

```python
transform = transforms.Compose([
    transforms.ToTensor(),                                   # [0, 255] → [0, 1], HWC → CHW
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # [0, 1]   → [-1, 1]
])
```

| Step | Effect |
|------|--------|
| `ToTensor()` | Converts PIL images to tensors and scales pixels from `0–255` to `0–1` |
| `Normalize(0.5, 0.5)` | Applies `(x − 0.5) / 0.5` per channel, mapping values to the range `[-1, 1]` |

**Batching:** `DataLoader` with batch size **64**. The training set is shuffled every epoch.

---

## 🧠 Model Architecture

```
Input                     3 × 32 × 32
  │
  ├─ Conv2d(3→32, 3×3, pad=1) + ReLU     →  32 × 32 × 32
  ├─ MaxPool2d(2×2)                      →  32 × 16 × 16
  │
  ├─ Conv2d(32→64, 3×3, pad=1) + ReLU    →  64 × 16 × 16
  ├─ MaxPool2d(2×2)                      →  64 ×  8 ×  8
  │
  ├─ Conv2d(64→128, 3×3, pad=1) + ReLU   → 128 ×  8 ×  8
  ├─ MaxPool2d(2×2)                      → 128 ×  4 ×  4
  │
  ├─ Flatten                             → 2048
  ├─ Linear(2048→256) + ReLU             → 256
  └─ Linear(256→10)                      → 10 class scores (logits)
```

**Design pattern:** each block **doubles the filters** (32 → 64 → 128) while max-pooling **halves the spatial size** (32 → 16 → 8 → 4). The network trades spatial resolution for richer feature representations. `padding=1` keeps the size unchanged through each convolution.

**Parameter breakdown**

| Layer | Parameters |
|-------|-----------:|
| Conv2d (3 → 32) | 896 |
| Conv2d (32 → 64) | 18,496 |
| Conv2d (64 → 128) | 73,856 |
| Linear (2048 → 256) | 524,544 |
| Linear (256 → 10) | 2,570 |
| **Total** | **620,362** |

---

## 🔁 Training

| Setting | Value |
|---------|-------|
| Loss function | `CrossEntropyLoss` |
| Optimizer | Adam (default learning rate 0.001) |
| Batch size | 64 |
| Epochs | 10 |

**Training loss per epoch**

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|---|---|---|---|---|---|---|---|---|---|
| **Loss** | 1.360 | 0.927 | 0.746 | 0.610 | 0.503 | 0.407 | 0.320 | 0.249 | 0.187 | **0.147** |

The training loss falls steadily, from **1.36 to 0.15**, a ~89% reduction over 10 epochs.

---

## 📈 Results

Evaluated on the full **10,000-image test set** with gradients disabled (`torch.no_grad()`):

```
Total test images : 10,000
Correct           :  7,476
Accuracy          : 74.76%
```

The model is **~7.5× better than random guessing** (10%), using a simple architecture with no data augmentation, regularization or pretrained weights.

---

## 💡 Key Insights

- **A simple CNN goes a long way.** Three conv blocks and no extra techniques reach ~75% accuracy on CIFAR-10.
- **Signs of overfitting.** Training loss keeps dropping sharply (0.147 by epoch 10) while test accuracy sits at ~75%. The model is starting to memorize the training data. Dropout, data augmentation and a validation set would help here.
- **The parameters sit in the classifier, not the conv layers.** The first fully connected layer (`2048 → 256`) holds **~85% of all parameters**, while the three conv layers together hold only ~15%. Global average pooling or a smaller dense layer could shrink the model a lot.

---

## 🗂️ Project Structure

```
CNN_CIFAR10_Image_Classfication/
├── cnn_implementation.ipynb   # Data loading, CNN definition, training and evaluation
└── README.md
```

> The `data/` folder created by `torchvision` should be kept out of Git (add `data/` to `.gitignore`).

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ajstyle920/minor-ai-projects.git
cd minor-ai-projects/CNN_CIFAR10_Image_Classfication

# 2. Install dependencies
pip install torch torchvision jupyter

# 3. Launch the notebook (CIFAR-10 downloads automatically on first run)
jupyter notebook cnn_implementation.ipynb
```

> 💡 Training runs on CPU by default. Moving the model and batches to a GPU (`.to("cuda")` or `.to("mps")`) speeds it up a lot.

---

## 🔮 Limitations & Future Work

- [ ] **Validation split.** Carve a validation set out of the training data to monitor overfitting during training.
- [ ] **Data augmentation:** random crops, horizontal flips and colour jitter.
- [ ] **Regularization:** Dropout and Batch Normalization.
- [ ] **Learning rate scheduling** (e.g. `StepLR`, `CosineAnnealingLR`).
- [ ] **GPU / MPS support** for faster training.
- [ ] **Per-class accuracy and a confusion matrix** to see which classes get confused (e.g. cat vs dog).
- [ ] **Training curves:** loss and accuracy plots.
- [ ] **Save the trained model** with `torch.save()` for reuse.
- [ ] **Transfer learning** with a pretrained ResNet, for comparison.

---

## 👤 Author

**Md Altaf Jawed**
BS in Data Science and Applications, **IIT Madras**

[![GitHub](https://img.shields.io/badge/GitHub-ajstyle920-181717?logo=github)](https://github.com/ajstyle920)

> Part of the [Minor AI Projects](https://github.com/ajstyle920/minor-ai-projects) collection.
