# SmartCart Customer Segmentation System

## Project Overview

SmartCart Customer Segmentation System is an unsupervised machine learning project that identifies distinct customer groups based on purchasing behavior, demographics, spending patterns, and engagement metrics.

The goal is to help businesses understand their customers better and design targeted marketing strategies for each customer segment.

---

## Problem Statement

Businesses often treat all customers the same, resulting in inefficient marketing campaigns and lower customer engagement.

The objective of this project is to segment customers into meaningful groups using clustering techniques so that marketing efforts can be personalized according to customer behavior and spending habits.

---

## Dataset Information

The dataset contains customer demographic and purchasing information, including:

* Income
* Education Level
* Marital Status
* Product Purchases
* Number of Children
* Customer Registration Date
* Campaign Responses
* Website Visits
* Recency of Purchases

These attributes provide valuable insights into customer purchasing patterns and engagement levels.

---

## Data Preprocessing

### Missing Value Treatment

Missing values in the Income column were handled using median imputation to preserve the overall distribution of customer income.

### Outlier Removal

Outliers were identified through exploratory analysis and removed to improve clustering quality.

The following anomalies were filtered:

* Extremely high income values
* Unrealistic customer ages

Removing these observations helped create more meaningful customer segments.

---

## Feature Engineering

Several new features were created to improve business understanding and clustering performance.

### Age

Customer age was calculated using:

```python
Age = Current Year - Year_Birth
```

### Customer Tenure

Customer registration dates were converted into the number of days since enrollment.

```python
Customer_days
```

This represents how long a customer has been associated with the business.

### Total Spending

A consolidated spending feature was created by combining purchases across product categories:

* Wines
* Fruits
* Meat Products
* Fish Products
* Sweet Products
* Gold Products

This provided a single measure of overall customer value.

### Total Children

A household size indicator was created:

```python
Total_children = Kidhome + Teenhome
```

---

## Category Simplification

To improve interpretability, categorical variables were simplified.

### Education Categories

Original education levels were grouped into:

* Undergraduate
* Graduate
* Postgraduate

### Living Status

Marital categories were transformed into:

* Partner
* Alone

This created more meaningful business-focused customer profiles.

---

## Exploratory Data Analysis (EDA)

Several analyses were performed to understand customer behavior.

### Pairwise Relationship Analysis

Pair plots were used to explore relationships among:

* Income
* Spending
* Customer Response
* Age
* Number of Children

### Correlation Analysis

A correlation heatmap was generated to identify:

* Spending drivers
* Income relationships
* Customer behavior patterns

Key insight:

Income showed a strong positive relationship with total customer spending.

---

## Data Encoding

Categorical features were converted into numerical format using:

### One-Hot Encoding

Applied to:

* Education
* Living Status

This ensured compatibility with machine learning algorithms while preserving categorical information.

---

## Feature Scaling

All numerical variables were standardized using:

* StandardScaler

Scaling ensured that features contributed equally during clustering.

---

## Dimensionality Reduction

### Principal Component Analysis (PCA)

PCA was used to reduce dimensionality and visualize customer groups.

#### 2-Dimensional PCA

Provided an initial visualization of customer distribution.

#### 3-Dimensional PCA

Captured a larger percentage of dataset variance and enabled better cluster visualization.

---

## Customer Segmentation

### K-Means Clustering

K-Means was used to identify customer segments.

### Optimal Cluster Selection

The Elbow Method was applied to determine the optimal number of clusters.

Using Within Cluster Sum of Squares (WCSS), the optimal number of clusters was found to be:

```text
K = 4
```

---

## Cluster Visualization

Customer segments were visualized using:

* 2D PCA Scatter Plots
* 3D PCA Scatter Plots

Color-coded clusters made customer group separation easier to interpret.

---

## Cluster Insights

### Cluster 0

* Lower income customers
* Lower spending behavior
* Larger households

### Cluster 1

* Moderate income customers
* Moderate to high spending
* Consistent purchasing activity

### Cluster 2

* Highest income customers
* Highest spending levels
* Strong campaign responsiveness
* High-value customer segment

### Cluster 3

* Lower income customers
* Low spending behavior
* Similar characteristics to Cluster 0

---

## Business Recommendations

### Premium Marketing Campaigns

Target Cluster 2 with:

* Premium products
* Loyalty rewards
* Exclusive offers

### Growth Campaigns

Target Cluster 1 with:

* Upselling opportunities
* Cross-selling strategies

### Budget-Friendly Promotions

Target Clusters 0 and 3 with:

* Discounts
* Value bundles
* Seasonal offers

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* KMeans Clustering
* PCA
* KneeLocator

---

## Key Learning Outcomes

This project demonstrates:

* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis
* Dimensionality Reduction
* Customer Segmentation
* Unsupervised Machine Learning
* Business Insight Generation

---

## Author

Altaf Jawed

Customer Segmentation and Unsupervised Machine Learning Project focused on identifying actionable customer groups for targeted marketing strategies.
