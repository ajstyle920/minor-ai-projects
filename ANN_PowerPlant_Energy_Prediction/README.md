# ⚡ ANN Power Plant Energy Prediction

Predicting the **net electrical energy output** of a Combined Cycle Power Plant from ambient conditions. The model is an **Artificial Neural Network built from scratch in PyTorch**, and it is benchmarked against classical machine learning regressors.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Task](https://img.shields.io/badge/Task-Regression-blue)
![R²](https://img.shields.io/badge/Test%20R%C2%B2-0.934-success)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Results at a Glance](#-results-at-a-glance)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
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

A combined cycle power plant's output depends heavily on environmental conditions. This project trains a small feed-forward neural network to learn that relationship from four ambient variables and predict the plant's energy output.

The project is built as a **deep learning fundamentals exercise**. It covers the full PyTorch workflow: converting data to tensors, `TensorDataset` and `DataLoader` batching, defining a custom `nn.Module`, writing the training and validation loop by hand, and checkpointing the best model.

---

## 🏆 Results at a Glance

| Model | Test R² | Test MSE |
|-------|:-------:|:--------:|
| **ANN (PyTorch)** | **0.934** | **18.88** |
| Support Vector Regressor (SVR) | 0.943 | – |
| Random Forest Regressor | _TBD_ | – |

> The ANN predicts energy output with an average error of about **±4.3 MW** (RMSE ≈ √18.88).

---

## 🎯 Problem Statement

**Given** four ambient conditions measured at the plant:

| Feature | Description | Unit |
|---------|-------------|------|
| `AT` | Ambient Temperature | °C |
| `V`  | Exhaust Vacuum | cm Hg |
| `AP` | Ambient Pressure | millibar |
| `RH` | Relative Humidity | % |

**Predict:**

| Target | Description | Unit |
|--------|-------------|------|
| `PE` | Net hourly Electrical Energy Output | MW |

This is a **supervised regression** problem.

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository, Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)
- **File used:** `powerplant_data.csv`
- **Features:** 4 numerical inputs (`AT`, `V`, `AP`, `RH`)
- **Target:** 1 numerical output (`PE`)
- **Missing values:** none

**Sample rows**

| AT | V | AP | RH | PE |
|---:|---:|---:|---:|---:|
| 8.34 | 40.77 | 1010.84 | 90.01 | 480.48 |
| 23.64 | 58.49 | 1011.40 | 74.20 | 445.75 |
| 29.74 | 56.90 | 1007.15 | 41.91 | 438.76 |
| 19.07 | 49.69 | 1007.22 | 76.79 | 453.09 |
| 11.80 | 40.66 | 1017.13 | 97.20 | 464.43 |

> ⚠️ The dataset is **not included** in this repository. Download it from the link above and save it as `powerplant_data.csv` in this folder.

---

## 🔧 Methodology

```
CSV ──▶ Missing-value check ──▶ Feature/target split ──▶ Train/Test split (80/20)
    ──▶ StandardScaler ──▶ PyTorch tensors ──▶ TensorDataset ──▶ DataLoader (batch 32)
    ──▶ ANN training (100 epochs) ──▶ Best-model checkpoint ──▶ Evaluation ──▶ ML comparison
```

1. **Load and inspect** the data with Pandas and confirm there are no missing values.
2. **Split** the data into features (`AT`, `V`, `AP`, `RH`) and target (`PE`).
3. **Train/test split** of 80/20 with `random_state=42`.
4. **Standardize** the features with `StandardScaler`, fitted on the training set only.
5. **Convert to tensors.** The target is reshaped to `(N, 1)` with `.view(-1, 1)`.
6. **Batch** the data with `TensorDataset` and `DataLoader` (batch size 32, shuffled for training).

---

## 🧠 Model Architecture

```
Input (4 features)
      │
 Linear(4 → 6) ── ReLU        # Hidden layer 1
      │
 Linear(6 → 6) ── ReLU        # Hidden layer 2
      │
 Linear(6 → 1)                # Output (predicted PE)
```

| Component | Choice |
|-----------|--------|
| Hidden layers | 2 × 6 neurons |
| Activation | ReLU |
| Output | 1 neuron (linear, for regression) |
| Trainable parameters | **79** |
| Loss function | Mean Squared Error (`nn.MSELoss`) |
| Optimizer | Adam (default learning rate 0.001) |
| Epochs | 100 |
| Batch size | 32 |

```python
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 6), nn.ReLU(),
            nn.Linear(6, 6), nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        return self.model(x)
```

---

## 🔁 Training

Each epoch runs:

1. **Training mode** (`model.train()`): forward pass → MSE loss → `loss.backward()` → `optimizer.step()`, once per mini-batch.
2. **Validation mode** (`model.eval()` + `torch.no_grad()`): compute the loss on held-out data without updating weights.
3. **Checkpointing:** whenever validation loss reaches a new minimum, the weights are saved to `best_model.pt`.

**Loss progression (MSE)**

| Epoch | Train Loss | Validation Loss |
|------:|-----------:|----------------:|
| 1 | 205,522 | 202,387 |
| 5 | 38,224 | 27,829 |
| 10 | 6,456 | 5,186 |
| 20 | 339 | 304 |
| 30 | 32.3 | 28.2 |
| 40 | 21.6 | 19.6 |
| 93 ⭐ | 21.1 | **18.87** (best) |
| 100 | 20.7 | 18.93 |

The loss falls steeply in the first ~30 epochs and levels off around **MSE ≈ 19–21** from epoch 40 onward.

---

## 📈 Results

**ANN (best checkpoint, epoch 93)**

| Metric | Train | Test |
|--------|:-----:|:----:|
| MSE | 20.69 | 18.88 |
| RMSE | 4.55 | 4.34 |
| R² | – | **0.934** |

**Comparison with classical ML**

| Model | Configuration | Test R² |
|-------|---------------|:-------:|
| ANN (PyTorch) | 4 → 6 → 6 → 1, ReLU, Adam | 0.934 |
| SVR | Default (RBF kernel), scaled features | **0.943** |
| Random Forest | 201 trees, `max_depth=3`, OOB enabled | _TBD_ |

---

## 💡 Key Insights

- **Strong fit with a tiny network.** Only 79 parameters explain about 93% of the variance in energy output.
- **No overfitting.** Test MSE (18.88) is slightly *lower* than train MSE (20.69), so the model generalizes well.
- **Classical ML is competitive.** SVR slightly outperforms the ANN (0.943 vs 0.934). On a small tabular dataset with 4 features, a neural network doesn't automatically beat well-established ML methods.
- **Very high initial loss.** Training starts at MSE ≈ 205,000 because the target (`PE` ≈ 420–495 MW) is **not scaled**. The network has to learn a large output bias first, which slows the early epochs.

---

## 🗂️ Project Structure

```
ANN_PowerPlant_Energy_Prediction/
├── ANN_PowerPlant_Energy_Prediction.ipynb   # Full pipeline: preprocessing, ANN, evaluation, ML comparison
├── best_model.pt                            # Best ANN weights (lowest validation loss)
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ajstyle920/minor-ai-projects.git
cd minor-ai-projects/ANN_PowerPlant_Energy_Prediction

# 2. Install dependencies
pip install torch pandas numpy scikit-learn jupyter

# 3. Add the dataset as powerplant_data.csv in this folder

# 4. Launch the notebook
jupyter notebook ANN_PowerPlant_Energy_Prediction.ipynb
```

**Loading the saved model**

```python
model = ANN()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
```

> Inputs must be scaled with the same `StandardScaler` fitted on the training data. Re-run the preprocessing cells before making predictions.

---

## 🔮 Limitations & Future Work

- [ ] **Separate validation set.** The test set is currently also used for validation and checkpoint selection. A dedicated train/validation/test split would give a fully unbiased test score.
- [ ] **Scale the target variable** to speed up convergence and stabilize early training.
- [ ] **Save the fitted scaler** (e.g. with `joblib`) so the model can be used for inference on its own.
- [ ] **Hyperparameter tuning:** layer sizes, learning rate, batch size.
- [ ] **Early stopping**, since the loss plateaus after ~40 epochs.
- [ ] **Loss curve and predicted-vs-actual plots** with Matplotlib.
- [ ] **K-fold cross-validation** for more reliable model comparison.
- [ ] **Experiment tracking** with TensorBoard.

---

## 👤 Author

**Md Altaf Jawed**
BS in Data Science and Applications, **IIT Madras**

[![GitHub](https://img.shields.io/badge/GitHub-ajstyle920-181717?logo=github)](https://github.com/ajstyle920)

> Part of the [Minor AI Projects](https://github.com/ajstyle920/minor-ai-projects) collection.
