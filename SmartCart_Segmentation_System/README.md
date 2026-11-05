# 🛒 SmartCart: Customer Segmentation System

An **unsupervised machine learning** project that groups **2,236 retail customers into 4 distinct segments** based on income, spending, purchase channels, family structure and campaign engagement. It uses **K-Means clustering on PCA-reduced features** and turns each segment into an actionable marketing profile.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Task](https://img.shields.io/badge/Task-Clustering-blue)
![Segments](https://img.shields.io/badge/Segments-4-success)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Segments at a Glance](#-segments-at-a-glance)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Data Cleaning](#-data-cleaning)
- [Correlation Analysis](#-correlation-analysis)
- [Encoding & Scaling](#-encoding--scaling)
- [Dimensionality Reduction (PCA)](#-dimensionality-reduction-pca)
- [Clustering (K-Means)](#-clustering-k-means)
- [Segment Profiles](#-segment-profiles)
- [Marketing Recommendations](#-marketing-recommendations)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Limitations & Future Work](#-limitations--future-work)
- [Author](#-author)

---

## 🔍 Overview

Not all customers are the same. Treating them as one group wastes marketing budget and misses opportunities. **SmartCart** uses customer data to find natural groupings, with **no predefined labels**, so a business can tailor campaigns, offers and channels to each segment.

The project covers the full unsupervised learning workflow: missing-value handling, feature engineering, category simplification, outlier removal, correlation analysis, one-hot encoding, standardization, PCA, choosing *k* with the elbow method, K-Means clustering and cluster profiling.

---

## 🏆 Segments at a Glance

| Cluster | Segment | Size | Avg Income | Avg Spending | Campaign Response |
|:-------:|---------|:----:|:----------:|:------------:|:-----------------:|
| **2** | 💎 Premium Spenders | 642 (28.7%) | **$74.7K** | **$1,306** | **23.8%** |
| **1** | 🎓 Established Deal-Seekers | 426 (19.1%) | $59.9K | $796 | 17.4% |
| **3** | 🧍 Budget Singles | 458 (20.5%) | $38.1K | $183 | 13.8% |
| **0** | 👨‍👩‍👧 Budget Families | 710 (31.8%) | $35.6K | $131 | 6.2% |

> 💡 **Premium Spenders spend ~10× more than Budget Families** and respond to campaigns ~4× as often.

---

## 💼 Business Problem

A retailer wants to answer questions like:

- *Who are our most valuable customers?*
- *Which customers respond to marketing campaigns?*
- *Which channels (web, catalog, store) does each group prefer?*
- *Who is price-sensitive and driven by discounts?*

Customer segmentation answers these questions. It replaces one-size-fits-all marketing with **targeted strategies for each segment**.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `smart_cart.csv` |
| Raw shape | 2,240 customers × 22 columns |
| After cleaning | 2,236 customers |
| Missing values | Only in `Income` (filled with the median) |

**Raw features**

| Category | Columns |
|----------|---------|
| 👤 **Demographics** | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| 📅 **Customer history** | `Dt_Customer` (enrolment date), `Recency` (days since last purchase), `Complain` |
| 🛍️ **Spending (2 years)** | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| 🛒 **Purchase channels** | `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumDealsPurchases`, `NumWebVisitsMonth` |
| 📣 **Campaign** | `Response` (accepted the last campaign: 1/0) |

> ⚠️ The dataset is **not included** in this repository. Place `smart_cart.csv` in this folder before running the notebook.

---

## 🔧 Pipeline

```
Raw CSV (2240 × 22)
   │
   ├─▶ Fill missing Income (median)
   ├─▶ Feature engineering (Age, Customer_days, Total_spending, Total_children)
   ├─▶ Simplify categories (Education → 3 levels, Marital_Status → Living_With)
   ├─▶ Drop redundant columns ─────────────────────────▶ 2240 × 15
   ├─▶ Remove outliers (Age ≥ 90, Income ≥ 600K) ──────▶ 2236 × 15
   ├─▶ Correlation analysis (heatmap)
   ├─▶ One-hot encoding ────────────────────────────────▶ 2236 × 18
   ├─▶ StandardScaler
   ├─▶ PCA (3 components, ~45% variance)
   ├─▶ Elbow method + KneeLocator ──▶ k = 4
   ├─▶ K-Means (k = 4)
   └─▶ Cluster profiling & visualization
```

---

## 🛠️ Feature Engineering

| New Feature | Formula | Purpose |
|-------------|---------|---------|
| `Age` | `2026 − Year_Birth` | Age is easier to interpret than birth year |
| `Customer_days` | `today − Dt_Customer` | Customer tenure (loyalty) in days |
| `Total_spending` | Sum of all 6 `Mnt*` columns | One measure of customer value |
| `Total_children` | `Kidhome + Teenhome` | Household size / family responsibility |

**Category simplification**

| Original | Simplified | Counts |
|----------|-----------|--------|
| `Education`: Basic, 2n Cycle, Graduation, Master, PhD | **Undergraduate** · **Graduate** · **Postgraduate** | 257 · 1,127 · 856 |
| `Marital_Status`: Married, Together, Single, Divorced, Widow, Alone, Absurd, YOLO | `Living_With`: **Partner** · **Alone** | 1,444 · 796 |

> Turning 8 noisy marital categories (including invalid ones like *"Absurd"* and *"YOLO"*) into a single **Partner / Alone** flag gives a cleaner signal of household structure.

**Dropped after engineering:** `ID`, `Year_Birth`, `Dt_Customer`, `Kidhome`, `Teenhome`, the 6 `Mnt*` columns and `Marital_Status`.

---

## 🧹 Data Cleaning

| Step | Action | Result |
|------|--------|--------|
| Missing values | `Income` filled with the **median** (robust to skew) | No nulls |
| Outliers | Identified with a **pairplot** of Income, Recency, Response, Age, Total_spending and Total_children | 4 extreme records found |
| Removal | Kept `Age < 90` and `Income < 600,000` | 2,240 → **2,236** rows |

---

## 🔗 Correlation Analysis

The strongest relationships in the correlation heatmap:

| Feature Pair | Correlation | Meaning |
|--------------|:-----------:|---------|
| Income ↔ Total_spending | **+0.79** | Higher earners spend much more |
| Total_spending ↔ Catalog purchases | **+0.78** | Big spenders buy through catalogs |
| Income ↔ Catalog purchases | +0.69 | Catalog is a premium channel |
| Total_spending ↔ Store purchases | +0.68 | In-store shopping drives value |
| Income ↔ Web visits/month | **−0.65** | Lower-income customers browse more but buy less |
| Total_spending ↔ Total_children | −0.50 | More children → lower discretionary spending |
| Total_children ↔ Deal purchases | +0.44 | Families are more discount-driven |

---

## 🔢 Encoding & Scaling

- **One-hot encoding** of `Education` (3 columns) and `Living_With` (2 columns), with all categories kept → **18 features**
- **StandardScaler** puts every feature on the same scale (mean 0, std 1). This is essential for both PCA and distance-based K-Means.

---

## 📉 Dimensionality Reduction (PCA)

PCA compresses the 18 correlated features into a few uncorrelated principal components, for visualization and clustering.

| Component | Explained Variance | Cumulative |
|-----------|:------------------:|:----------:|
| PC1 | 23.2% | 23.2% |
| PC2 | 11.4% | 34.5% |
| PC3 | 10.4% | **45.0%** |

**3 components** were kept (~45% of the variance), which allows both 2D and 3D cluster visualizations.

---

## 🎯 Clustering (K-Means)

**Choosing k: the elbow method**

- WCSS (inertia) was computed for **k = 1 to 9** on the PCA-reduced data
- The elbow was detected automatically with **`KneeLocator`** (`curve="convex"`, `direction="decreasing"`)
- **Optimal k = 4**

**Final model:** `KMeans(n_clusters=4, random_state=42)` fitted on the 3 principal components.

**Visualizations produced**
- 2D PCA scatter plot coloured by cluster
- 3D PCA projection coloured by cluster
- Cluster size count plot
- Income vs Total Spending scatter plot coloured by cluster

---

## 👥 Segment Profiles

**Cluster means**

| Metric | C0 · Budget Families | C1 · Established Deal-Seekers | C2 · Premium Spenders | C3 · Budget Singles |
|--------|:---:|:---:|:---:|:---:|
| **Customers** | 710 | 426 | 642 | 458 |
| **Income** | $35,605 | $59,857 | **$74,671** | $38,097 |
| **Total Spending** | $131 | $796 | **$1,306** | $183 |
| **Children** | 1.25 | 1.10 | **0.32** | 1.24 |
| **Age** | 54.4 | **62.4** | 57.7 | 55.6 |
| **Deal Purchases** | 2.2 | **3.6** | 1.5 | 2.5 |
| **Web Purchases** | 2.4 | **6.3** | 5.4 | 2.8 |
| **Catalog Purchases** | 0.6 | 3.5 | **5.7** | 0.9 |
| **Store Purchases** | 3.5 | 7.6 | **8.6** | 3.8 |
| **Web Visits / Month** | **6.5** | 5.6 | 3.0 | **6.5** |
| **Campaign Response** | 6.2% | 17.4% | **23.8%** | 13.8% |
| **Living with Partner** | 100% | 82% | 60% | 0% |
| **Postgraduate** | 27% | **83%** | 24% | 34% |

### 💎 Cluster 2: Premium Spenders (642 customers)
The **highest income and highest spending**, with **few children**. They buy heavily through **catalog and store** channels, rarely need discounts, visit the website least, and have the **best campaign response rate (23.8%)**. This is the most valuable segment.

### 🎓 Cluster 1: Established Deal-Seekers (426 customers)
**Upper-middle income**, the **oldest** group (avg. 62) and **highly educated** (83% postgraduate). They spend well and are the **most active online buyers**, but also make the **most deal purchases**. They are valuable but price-aware.

### 🧍 Cluster 3: Budget Singles (458 customers)
**Lower income, low spending**, and **all living alone**, with a similar number of children to Cluster 0. They browse the website often but buy little. Their campaign response (13.8%) is still more than **double** that of Budget Families.

### 👨‍👩‍👧 Cluster 0: Budget Families (710 customers, the largest segment)
**The lowest income and lowest spending**. They **all live with a partner** and have the most children (1.25). They browse online often but convert rarely, and have the **lowest campaign response (6.2%)**. Family costs probably limit their discretionary spending.

---

## 📣 Marketing Recommendations

| Segment | Strategy | Channels |
|---------|----------|----------|
| 💎 **Premium Spenders** | Loyalty and VIP programs, premium product launches, early access, personalized recommendations. Focus on **retention**. | Catalog, in-store |
| 🎓 **Established Deal-Seekers** | Targeted discounts and bundles on premium items, email offers. **Upsell** with deal-driven campaigns. | Web, email |
| 🧍 **Budget Singles** | Small-pack offers, single-person bundles, re-engagement campaigns. They respond better than expected. | Web, mobile |
| 👨‍👩‍👧 **Budget Families** | Family packs, bulk discounts, kids' products, coupons. Turn browsing into purchases with **web-to-cart nudges**. | Web, deals |

---

## 💡 Key Insights

1. **Income drives spending.** Income and total spending are strongly correlated (r = 0.79), and the clusters line up along this axis.
2. **The top segment is ~10× more valuable.** Premium Spenders average $1,306 against $131 for Budget Families.
3. **Children reduce discretionary spending.** The highest-spending cluster has the fewest children (0.32 on average).
4. **High web traffic ≠ high value.** The two lowest-spending clusters visit the website the most (6.5 visits/month) but purchase the least. This is a conversion opportunity.
5. **Campaigns work best on Premium Spenders.** Their response rate is 23.8%, against 6.2% for Budget Families.
6. **Household structure splits the budget customers.** Clusters 0 and 3 have similar income and spending but are separated perfectly by `Living_With` (100% partner vs 100% alone). Binary one-hot features carry a lot of weight after scaling.

---

## 🗂️ Project Structure

```
SmartCart_Segmentation_System/
├── SmartCart_Segmentation_System.ipynb   # Full pipeline: cleaning, feature engineering, PCA, K-Means, profiling
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ajstyle920/minor-ai-projects.git
cd minor-ai-projects/SmartCart_Segmentation_System

# 2. Install dependencies
pip install pandas matplotlib seaborn scikit-learn kneed jupyter

# 3. Add smart_cart.csv to this folder

# 4. Launch the notebook
jupyter notebook SmartCart_Segmentation_System.ipynb
```

---

## 🔮 Limitations & Future Work

**Known limitations**

- **Time-dependent features.** `Customer_days` is calculated from `pd.Timestamp.today()`, so its values change every time the notebook runs. Using the latest enrolment date in the dataset as the reference point would make results reproducible.
- **Information loss from PCA.** Clustering on 3 principal components keeps only ~45% of the variance.
- **The elbow method alone.** *k* was chosen with a single heuristic. No silhouette or other validity score was checked.
- **Campaign outcome used as a clustering feature.** `Response` and `Complain` are behavioural outcomes. Using them only to profile clusters afterwards, not to form them, could give cleaner segments.

**Planned improvements**

- [ ] Validate *k* with **silhouette score**, Davies–Bouldin and Calinski–Harabasz
- [ ] Compare clustering on the **full scaled feature set** against PCA-reduced data
- [ ] Try other algorithms: **Agglomerative (hierarchical)**, **DBSCAN**, **Gaussian Mixture Models**
- [ ] Add **RFM (Recency, Frequency, Monetary)** features
- [ ] Use a fixed reference date for `Customer_days` and `Age`
- [ ] Save the scaler, PCA and K-Means models for scoring new customers
- [ ] Build an interactive **dashboard** (Streamlit / Plotly) for exploring segments

---

## 👤 Author

**Md Altaf Jawed**
BS in Data Science and Applications, **IIT Madras**

[![GitHub](https://img.shields.io/badge/GitHub-ajstyle920-181717?logo=github)](https://github.com/ajstyle920)

> Part of the [Minor AI Projects](https://github.com/ajstyle920/minor-ai-projects) collection.
