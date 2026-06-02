# CreditWise Loan Approval Prediction System

## Project Overview

CreditWise is a Machine Learning project designed to predict whether a loan application should be approved based on an applicant's financial and demographic information.

The objective is to assist financial institutions in making faster and more consistent lending decisions by identifying patterns associated with successful loan approvals.

---

## Problem Statement

Loan approval is a critical decision-making process in banking and financial services. Incorrect approvals can increase financial risk, while incorrect rejections can lead to missed business opportunities.

This project aims to build a classification model capable of predicting loan approval outcomes using applicant information such as income, credit score, employment status, debt-to-income ratio, and other relevant attributes.

---

## Dataset Information

The dataset contains approximately:

* 1,000 loan applications
* 19 input features
* 1 target variable (`Loan_Approved`)

Features include:

* Applicant Income
* Coapplicant Income
* Credit Score
* Employment Status
* Education Level
* Marital Status
* Property Area
* Loan Purpose
* Debt-to-Income Ratio (DTI)
* Gender
* Employer Category

Target Variable:

* Loan Approved (Yes/No)

---

## Data Preprocessing

### Missing Value Treatment

Missing values were handled using Scikit-Learn's `SimpleImputer`.

* Numerical features → Mean Imputation
* Categorical features → Most Frequent Value Imputation

This ensured that no records were lost due to missing data.

---

## Exploratory Data Analysis (EDA)

Several exploratory analyses were performed to better understand the dataset:

### Class Distribution Analysis

* Loan approval distribution visualized using pie charts.
* Checked whether the target classes were balanced.

### Income Analysis

* Applicant Income Distribution
* Coapplicant Income Distribution

Histogram visualizations were used to identify income patterns.

### Categorical Feature Analysis

* Gender Distribution
* Employment Characteristics

Bar charts were used for category analysis.

### Outlier Detection

Boxplots were created to inspect:

* Applicant Income
* Credit Score
* Other important numerical variables

### Credit Score Analysis

Examined the relationship between:

* Credit Score
* Loan Approval Status

This helped identify the importance of credit score in approval decisions.

---

## Feature Encoding

Since machine learning models require numerical inputs, categorical features were transformed using:

### Label Encoding

Applied to ordinal variables such as:

* Education Level
* Loan Approval

### One-Hot Encoding

Applied to nominal categorical variables including:

* Employment Status
* Marital Status
* Loan Purpose
* Property Area
* Gender
* Employer Category

This converted categorical information into machine-readable numerical features.

---

## Correlation Analysis

A correlation heatmap was generated to:

* Understand relationships among numerical variables.
* Identify highly influential features.
* Detect multicollinearity.

Correlation scores were also analyzed against the target variable to determine feature importance.

---

## Model Development

The dataset was divided into:

* Training Set: 80%
* Testing Set: 20%

Feature scaling was performed using:

* StandardScaler

to ensure fair comparison across models.

### Models Implemented

#### 1. Logistic Regression

A baseline classification model used for binary loan approval prediction.

#### 2. K-Nearest Neighbors (KNN)

Distance-based classification algorithm used for comparison.

#### 3. Gaussian Naive Bayes

Probabilistic classifier assuming feature independence.

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy Score
* Precision Score
* Recall Score
* Confusion Matrix

Special attention was given to:

### Precision

Reducing false approvals of high-risk applicants.

### Recall

Reducing rejection of potentially good borrowers.

---

## Feature Engineering

Additional features were created to improve predictive performance:

### Squared Credit Score

```python
Credit_Score_sq = Credit_Score²
```

### Squared DTI Ratio

```python
DTI_Ratio_sq = DTI_Ratio²
```

These engineered features helped capture non-linear relationships within the dataset.

---

## Results

Multiple machine learning models were compared and evaluated.

Key observations:

* Credit Score was one of the strongest predictors of loan approval.
* Feature engineering improved model expressiveness.
* Gaussian Naive Bayes demonstrated strong performance in precision-based evaluation.
* Standardized features improved overall model stability.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Future Improvements

Potential enhancements include:

* Random Forest Classifier
* XGBoost
* Hyperparameter Optimization using GridSearchCV
* Feature Selection Techniques
* Model Explainability using SHAP
* Deployment as a Loan Approval Web Application

---

## Author

Altaf Jawed

Machine Learning Project focused on predictive analytics and loan approval classification.
