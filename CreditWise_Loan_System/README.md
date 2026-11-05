# 💳 CreditWise: Loan Approval Prediction System

An end-to-end **machine learning classification project** that predicts whether a loan application will be **approved or rejected** from an applicant's financial, employment and demographic profile. Three classifiers are built and compared, with model selection driven by **business-relevant metrics** rather than accuracy alone.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-blue)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-87%25-success)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Results at a Glance](#-results-at-a-glance)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Preprocessing & Encoding](#-preprocessing--encoding)
- [Modeling](#-modeling)
- [Results](#-results)
- [Feature Engineering Experiment](#-feature-engineering-experiment)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Limitations & Future Work](#-limitations--future-work)
- [Author](#-author)

---

## 🔍 Overview

Lenders have to make fast, consistent decisions on loan applications. **CreditWise** learns patterns from 1,000 past applications to predict approval outcomes. The workflow covers missing-value imputation, EDA, categorical encoding, correlation analysis, model comparison and a feature engineering experiment.

---

## 🏆 Results at a Glance

| Model | Accuracy | Precision | Recall | F1 |
|-------|:--------:|:---------:|:------:|:--:|
| **Logistic Regression** | **0.870** | 0.787 | **0.787** | **0.787** |
| Gaussian Naive Bayes | 0.865 | **0.804** | 0.738 | 0.769 |
| K-Nearest Neighbors (k=7) | 0.755 | 0.643 | 0.443 | 0.524 |

> 🥇 **Logistic Regression** gives the best overall balance. 🎯 **Naive Bayes** gives the highest precision, so it makes the fewest risky approvals.

---

## 💼 Business Problem

A loan approval model can make two kinds of mistakes, and each has a different cost:

| Error | What happens | Business cost | Metric to watch |
|-------|--------------|---------------|-----------------|
| **False Positive** | A high-risk applicant is **approved** | Default risk and financial loss | **Precision** ↑ |
| **False Negative** | A creditworthy applicant is **rejected** | Lost customer and lost revenue | **Recall** ↑ |

Because the classes are imbalanced (~70% rejected), **accuracy alone is misleading**. A model that rejects everyone would still score ~70%. The models are therefore compared mainly on **precision and recall**.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `loan_approval_data.csv` |
| Rows | 1,000 applications |
| Columns | 20 (`Applicant_ID` + 18 features + 1 target) |
| Missing values | 50 per column (~5%) |
| Target | `Loan_Approved` (Yes / No) |
| Class balance | **No: 702 (70.2%)** · **Yes: 298 (29.8%)** |

**Features**

| Type | Features |
|------|----------|
| 💰 **Financial** | `Applicant_Income`, `Coapplicant_Income`, `Savings`, `Collateral_Value`, `Existing_Loans`, `DTI_Ratio`, `Credit_Score` |
| 🏦 **Loan details** | `Loan_Amount`, `Loan_Term`, `Loan_Purpose` |
| 💼 **Employment** | `Employment_Status`, `Employer_Category` |
| 👤 **Demographic** | `Age`, `Gender`, `Marital_Status`, `Dependents`, `Education_Level`, `Property_Area` |

> ⚠️ The dataset is **not included** in this repository. Place `loan_approval_data.csv` in this folder before running the notebook.

---

## 🔧 Pipeline

```
Raw CSV (1000 × 20)
   │
   ├─▶ Missing-value imputation (SimpleImputer: mean / most-frequent)
   ├─▶ EDA (class balance, distributions, boxplots, credit-score analysis)
   ├─▶ Encoding (LabelEncoder + OneHotEncoder) ─▶ 1000 × 29
   ├─▶ Correlation analysis (heatmap + target correlations)
   ├─▶ Train/Test split (80/20, random_state=42) ─▶ 800 train / 200 test
   ├─▶ StandardScaler
   ├─▶ Models: Logistic Regression · KNN · Gaussian Naive Bayes
   └─▶ Feature engineering experiment (squared features)
```

---

## 📈 Exploratory Data Analysis

| Analysis | Visualization | Finding |
|----------|---------------|---------|
| Class balance | Pie chart | ~70% rejected and ~30% approved, so the classes are **imbalanced** |
| Gender distribution | Bar chart | Male 621 · Female 379 |
| Income distributions | Histograms | Applicant and co-applicant income spread |
| Outlier detection | Boxplots (Income, Credit Score, DTI, Savings vs approval) | No extreme outliers that needed removing |
| Credit score vs approval | Grouped histogram | **Higher credit scores are strongly associated with approval** |

**Correlation with `Loan_Approved`**

| Feature | Correlation | Interpretation |
|---------|:-----------:|----------------|
| `Credit_Score` | **+0.451** | Higher score → more likely approved |
| `DTI_Ratio` | **−0.445** | Higher debt burden → less likely approved |
| `Loan_Amount` | −0.126 | Larger loans → slightly less likely approved |
| `Applicant_Income` | +0.120 | Higher income → slightly more likely approved |
| `Loan_Term` | −0.087 | Longer terms → slightly less likely approved |

> **Credit Score** and **DTI Ratio** are by far the two strongest predictors. Every other feature has |r| < 0.13.

---

## 🧹 Preprocessing & Encoding

**Missing values** (`SimpleImputer`)

| Column type | Strategy |
|-------------|----------|
| Numerical | Mean |
| Categorical | Most frequent (mode) |

**Encoding**

| Encoder | Applied to | Reason |
|---------|-----------|--------|
| `LabelEncoder` | `Education_Level`, `Loan_Approved` | Binary / ordinal columns |
| `OneHotEncoder(drop="first")` | `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category` | Nominal columns with no natural order. `drop="first"` avoids the dummy-variable trap |

The 6 nominal columns expand into **15 binary columns**, giving a final dataset of **1000 × 29**.

**Scaling:** `StandardScaler` is fitted on the training set only and applied to both sets.

---

## 🤖 Modeling

| Model | Why it was chosen |
|-------|-------------------|
| **Logistic Regression** | Strong, interpretable baseline for binary classification |
| **K-Nearest Neighbors** (k = 7) | Distance-based, non-parametric comparison |
| **Gaussian Naive Bayes** | Fast probabilistic classifier, works well on small datasets |

---

## 📊 Results

Evaluated on the **200-application test set** (139 rejected · 61 approved):

| Model | Accuracy | Precision | Recall | F1 | False Approvals (FP) | Missed Good Borrowers (FN) |
|-------|:--------:|:---------:|:------:|:--:|:---:|:---:|
| **Logistic Regression** | **0.870** | 0.787 | **0.787** | **0.787** | 13 | **13** |
| Gaussian Naive Bayes | 0.865 | **0.804** | 0.738 | 0.769 | **11** | 16 |
| KNN (k = 7) | 0.755 | 0.643 | 0.443 | 0.524 | 15 | 34 |

**Confusion matrices** (rows = actual, columns = predicted)

```
 Logistic Regression       Gaussian Naive Bayes        KNN (k=7)
        Pred 0  Pred 1           Pred 0  Pred 1           Pred 0  Pred 1
True 0    126     13     True 0    128     11     True 0    124     15
True 1     13     48     True 1     16     45     True 1     34     27
```

---

## 🧪 Feature Engineering Experiment

To capture possible non-linear effects of the two strongest predictors, squared features were created and **used in place of the originals**:

```python
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
```

**Gaussian Naive Bayes, before and after**

| Version | Accuracy | Precision | Recall | F1 | FP | FN |
|---------|:--------:|:---------:|:------:|:--:|:--:|:--:|
| Original features | **0.865** | 0.804 | **0.738** | **0.769** | 11 | **16** |
| Squared features | 0.855 | **0.808** | 0.689 | 0.743 | **10** | 19 |

**Outcome:** precision went up slightly (one fewer false approval), but recall and accuracy dropped (three more good borrowers rejected). The squared features **did not improve overall performance**. This is a useful negative result: within the observed range, squaring a feature keeps nearly the same ordering, so it adds little new information.

---

## 💡 Key Insights

1. **Credit Score and DTI Ratio drive approval decisions.** Their correlations (+0.45 and −0.44) are more than 3× stronger than any other feature's.
2. **Logistic Regression is the best all-round model.** It has the highest accuracy, recall and F1, and its precision and recall are equal.
3. **Naive Bayes is the most conservative model.** It has the highest precision and the fewest risky approvals, which suits a risk-averse lender.
4. **KNN struggles in high dimensions.** With 28 features, distances become less meaningful (the *curse of dimensionality*). KNN misses 34 of 61 good borrowers.
5. **Accuracy hides trade-offs.** Logistic Regression and Naive Bayes have almost the same accuracy (87.0% vs 86.5%) but make very different kinds of errors.
6. **Not every feature engineering idea helps.** Squaring the top features traded recall for a negligible precision gain.

---

## 🗂️ Project Structure

```
CreditWise_Loan_System/
├── CreditWise_Loan_System.ipynb   # Full pipeline: imputation, EDA, encoding, modeling, experiments
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ajstyle920/minor-ai-projects.git
cd minor-ai-projects/CreditWise_Loan_System

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Add loan_approval_data.csv to this folder

# 4. Launch the notebook
jupyter notebook CreditWise_Loan_System.ipynb
```

---

## 🔮 Limitations & Future Work

**Known limitations**

- **Target values were imputed.** The 50 rows with a missing `Loan_Approved` label were filled with the mode ("No"). Dropping these rows would be more reliable than guessing the labels.
- **`Applicant_ID` is used as a feature.** It is just an identifier with no predictive meaning and should be dropped.
- **Preprocessing happens before the train/test split.** Imputation and encoding are fitted on the full dataset, which leaks a little information into the test set. A scikit-learn `Pipeline` would fix this.
- **Sensitive attributes.** `Gender` and `Marital_Status` are used as features. A real lending model would need fairness review, because these attributes can introduce discriminatory bias.

**Planned improvements**

- [ ] Build a `Pipeline` + `ColumnTransformer` for leak-free preprocessing
- [ ] Handle class imbalance (`class_weight="balanced"`, SMOTE)
- [ ] Tune decision thresholds and add ROC-AUC / precision-recall curves
- [ ] Use cross-validation (`StratifiedKFold`) for more robust comparison
- [ ] Try tree-based models: Random Forest, XGBoost
- [ ] Tune hyperparameters with `GridSearchCV` (e.g. the best `k` for KNN)
- [ ] Explain the model with SHAP
- [ ] Deploy as a web app (Streamlit / FastAPI)

---

## 👤 Author

**Md Altaf Jawed**
BS in Data Science and Applications, **IIT Madras**

[![GitHub](https://img.shields.io/badge/GitHub-ajstyle920-181717?logo=github)](https://github.com/ajstyle920)

> Part of the [Minor AI Projects](https://github.com/ajstyle920/minor-ai-projects) collection.
