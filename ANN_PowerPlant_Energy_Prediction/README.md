# ANN Power Plant Energy Prediction

## Overview

This project implements an Artificial Neural Network (ANN) from scratch using PyTorch to predict the electrical energy output of a Combined Cycle Power Plant.

The model learns the relationship between environmental variables such as temperature, pressure, humidity, and vacuum conditions to estimate the plant's power generation.

This project was built as part of my deep learning learning journey to understand:

- Data preprocessing
- Tensor conversion
- DataLoader and batching
- Neural network architecture design
- Forward propagation
- Backpropagation
- Model training and validation
- Regression using Artificial Neural Networks

---

## Problem Statement

Given environmental conditions:

- AT → Ambient Temperature
- V → Exhaust Vacuum
- AP → Ambient Pressure
- RH → Relative Humidity

Predict:

- PE → Electrical Energy Output

This is a supervised regression problem.

---

## Dataset

Power Plant Energy Output Dataset

Features:

| Feature | Description |
|----------|-------------|
| AT | Ambient Temperature |
| V | Exhaust Vacuum |
| AP | Ambient Pressure |
| RH | Relative Humidity |

Target Variable:

| Target | Description |
|---------|------------|
| PE | Net Electrical Energy Output |

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded dataset using Pandas
2. Checked for missing values
3. Separated features and target variable
4. Train-Test Split (80-20)
5. Standardized features using StandardScaler
6. Converted NumPy arrays into PyTorch tensors
7. Created TensorDataset objects
8. Used DataLoader for mini-batch training

---

## ANN Architecture

Network Structure:

Input Layer (4 Features)
        ↓
Dense Layer (6 Neurons)
        ↓
ReLU Activation
        ↓
Dense Layer (6 Neurons)
        ↓
ReLU Activation
        ↓
Output Layer (1 Neuron)

Loss Function:
- Mean Squared Error (MSE)

Optimizer:
- Adam Optimizer

Epochs:
- 100

Batch Size:
- 32

---

## Training Workflow

1. Forward Propagation
2. Loss Calculation (MSE)
3. Backpropagation
4. Weight Updates using Adam Optimizer
5. Validation after each epoch
6. Best Model Checkpoint Saving

---

## Model Evaluation

The trained model was evaluated using:

- Mean Squared Error (MSE)
- R² Score

Performance was measured on both training and testing datasets.

---

## Additional Comparison

To compare ANN performance with traditional Machine Learning approaches, the following models were also trained:

### Support Vector Regressor (SVR)

- Feature scaled input
- Evaluated using R² score

### Random Forest Regressor

- 201 Trees
- Max Depth = 3
- Out-of-Bag Evaluation Enabled

This comparison helps understand the strengths and limitations of ANN-based regression against classical ML algorithms.

---

## Project Structure

```text
ANN_PowerPlant_Energy_Prediction/
│
├── ANN_PowerPlant_Energy_Prediction.ipynb
├── README.md
└── best_model.pt
```

---

## Key Concepts Practiced

- Deep Learning Fundamentals
- Regression using ANN
- PyTorch Model Building
- Tensor Operations
- Mini-Batch Training
- Model Checkpointing
- Hyperparameter Configuration
- Evaluation Metrics

---

## Future Improvements

- Hyperparameter tuning
- Deeper neural network architectures
- Dropout regularization
- Early stopping
- K-Fold Cross Validation
- Experiment tracking using TensorBoard

---

## Author

Altaf Jawed

Artificial Intelligence, Machine Learning, Deep Learning, and Applied AI Projects
