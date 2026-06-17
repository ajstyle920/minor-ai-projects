# CNN CIFAR-10 Image Classification

## Overview

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify images from the CIFAR-10 dataset.

The model learns hierarchical image features using convolutional and pooling layers and predicts one of ten object categories present in the image.

This project was built to gain hands-on experience with:

- Computer Vision
- Convolutional Neural Networks (CNNs)
- Image Preprocessing
- PyTorch Deep Learning Workflow
- Model Training and Evaluation

---

## Problem Statement

Given a color image of size:

32 × 32 × 3

Predict which of the following 10 classes the image belongs to:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

This is a multi-class image classification problem.

---

## Dataset

### CIFAR-10 Dataset

The CIFAR-10 dataset contains:

- 60,000 color images
- 10 image classes
- 50,000 training images
- 10,000 testing images

Image Size:

32 × 32 × 3 RGB

Dataset was loaded using:

```python
torchvision.datasets.CIFAR10
```

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy

---

## Data Preprocessing

The following preprocessing steps were applied:

### Image to Tensor Conversion

```python
transforms.ToTensor()
```

Converts image pixels from:

```text
0 - 255
```

to

```text
0 - 1
```

### Normalization

```python
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

Scales image values approximately into:

```text
-1 to +1
```

---

## CNN Architecture

### Feature Extraction Layers

```text
Input Image
(32 × 32 × 3)

↓

Conv2D
3 → 32 Filters
Kernel Size = 3×3

↓

ReLU

↓

MaxPooling
2×2

↓

Conv2D
32 → 64 Filters
Kernel Size = 3×3

↓

ReLU

↓

MaxPooling
2×2

↓

Conv2D
64 → 128 Filters
Kernel Size = 3×3

↓

ReLU

↓

MaxPooling
2×2
```

---

### Classification Layers

```text
Flatten

↓

Linear
2048 → 256

↓

ReLU

↓

Linear
256 → 10

↓

Class Prediction
```

---

## Training Configuration

### Loss Function

```python
CrossEntropyLoss()
```

### Optimizer

```python
Adam Optimizer
```

### Batch Size

```python
64
```

### Epochs

```python
10
```

---

## Training Workflow

1. Load CIFAR-10 Dataset
2. Apply Transformations
3. Create DataLoaders
4. Build CNN Architecture
5. Forward Propagation
6. Compute Loss
7. Backpropagation
8. Update Weights using Adam
9. Evaluate on Test Dataset

---

## Model Evaluation

The model was evaluated on the CIFAR-10 test set using:

### Accuracy Score

```python
correct_predictions / total_predictions
```

Evaluation was performed using:

```python
torch.no_grad()
```

to disable gradient calculations during inference.

---

## Project Structure

```text
CNN_CIFAR10_Image_Classification/
│
├── CNN_CIFAR10_Image_Classification.ipynb
└── README.md
```

---

## Key Concepts Practiced

- Convolution Operations
- Feature Maps
- Kernel Filters
- ReLU Activation
- Max Pooling
- Flattening
- Fully Connected Layers
- Multi-Class Classification
- Cross Entropy Loss
- Adam Optimization
- CNN Training Pipeline
- Model Evaluation

---

## Future Improvements

- Add Validation Dataset
- Hyperparameter Tuning
- Data Augmentation
- Dropout Regularization
- Learning Rate Scheduling
- Transfer Learning using ResNet
- Confusion Matrix Visualization
- Training and Validation Accuracy Curves

---

## Author

Altaf Jawed

Deep Learning • Computer Vision • Artificial Intelligence
