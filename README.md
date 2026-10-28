# DBSCAN Anomaly Detection in Financial Transactions

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An unsupervised machine learning and pattern recognition project developed for the **AIML Honors (Pattern Recognition & Applications)** curriculum. It evaluates whether **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** can detect fraudulent credit card transactions in high-dimensional feature space without label supervision during model training.

---

## Table of Contents
- [Executive Overview](#executive-overview)
- [Problem Statement & Motivation](#problem-statement--motivation)
- [End-to-End Pipeline Architecture](#end-to-end-pipeline-architecture)
- [Methodology](#methodology)
  - [1. Data Preprocessing & Scaling](#1-data-preprocessing--scaling)
  - [2. Dimensionality Reduction (PCA)](#2-dimensionality-reduction-pca)
  - [3. K-Distance Analysis](#3-k-distance-analysis)
  - [4. DBSCAN Clustering & Anomaly Flagging](#4-dbscan-clustering--anomaly-flagging)
- [Experimental Results & Parameter Tuning](#experimental-results--parameter-tuning)
  - [Baseline vs. Optimal Model](#baseline-vs-optimal-model)
  - [Multi-Cluster Experiment](#multi-cluster-experiment)
- [Key Insights & Trade-Offs](#key-insights--trade-offs)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [References](#references)

---

## Executive Overview

Financial fraud detection is characterized by extreme class imbalance (typically <0.2% of transactions are fraudulent) and non-stationary fraud patterns that evolve over time. Supervised models often struggle when new fraud mechanics emerge or when labeled training instances are scarce and delayed.

This project implements an **unsupervised anomaly detection framework**:
* **Labels are strictly isolated:** Fraud labels (`Class`) are excluded during all preprocessing, PCA fitting, and DBSCAN clustering. Labels are used solely for post-hoc evaluation.
* **Core Result:** Achieved **81.25% Recall** (capturing 13 out of 16 fraud cases in a 10,000-transaction test subset) while isolating candidate anomalies to only **1.51%** of overall transaction volume.
* **Tuning Impact:** Hyperparameter sensitivity analysis across 12 configurations improved precision by reducing false positive flags by **12%** compared to the baseline.

---

## Problem Statement & Motivation

> **Core Research Question:** *"Can DBSCAN identify potentially fraudulent financial transactions as density-based anomalies without using fraud labels during model training?"*

DBSCAN is uniquely suited for anomaly detection because:
1. **Arbitrary Cluster Shapes:** Unlike K-Means, it does not assume spherical clusters.
2. **Explicit Noise Labeling:** It does not force every point into a cluster. Data points lying in sparse regions of the feature space are explicitly labeled as noise (`-1`), serving as natural anomaly candidates.
3. **No Predefined Cluster Count:** The algorithm discovers the natural number of dense regions rather than forcing a predetermined $k$.

---

## End-to-End Pipeline Architecture

```text
+------------------------------------+
|  ULB Credit Card Fraud Dataset     | (284,807 transactions, 30 features)
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| Fixed Experimental Subset (10,000) | (9,984 legitimate, 16 fraudulent)
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| Feature / Label Separation         | Target 'Class' removed & stored for eval
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| RobustScaler Transformation        | Rescales heavy-tailed financial distributions
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| PCA Dimensionality Reduction       | Compresses 30 features -> 23 components
|                                    | (Retains 95.02% cumulative variance)
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| K-Distance Elbow Analysis (k=10)   | Guides selection of neighbourhood radius (eps)
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| DBSCAN Density-Based Clustering    | Dense regions -> Clusters (0, 1, ...)
|                                    | Sparse regions -> Noise (-1) Anomaly Candidates
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| Post-Hoc Evaluation Scoring        | Recall: 81.25% | Precision: 8.61% | F1: 0.1557
+------------------------------------+
```

---

## Methodology

### 1. Data Preprocessing & Scaling
Financial transaction amounts exhibit severe right-skewness and extreme outliers. Standard scaling (Z-score) is heavily distorted by such outliers. We employ **`RobustScaler`**, which centers data using the median and scales using the Interquartile Range (IQR):
$$\text{IQR} = Q_3 - Q_1$$
This preserves the anomalous nature of extreme transactions without compressing legitimate feature variance.

### 2. Dimensionality Reduction (PCA)
Nearest-neighbor computations in high dimensions suffer from distance concentration (curse of dimensionality). **Principal Component Analysis (PCA)** was fitted across the standardized features, reducing dimensionality from 30 to 23 components while preserving **95.02%** of total variance.

### 3. K-Distance Analysis
To select a principled range for the neighborhood radius $\epsilon$ (`eps`), sorted $k$-nearest neighbor distances were computed ($k=10$):
* **Median $k$-distance:** 2.2755 (majority of transactions cluster closely)
* **95th Percentile:** 6.3251
* **99th Percentile:** 13.2793
* This revealed an elbow between 6.0 and 9.0, defining our experimental grid.

### 4. DBSCAN Clustering & Anomaly Flagging
Points are categorized into Core, Border, and Noise:
$$\text{Noise Point: } \{p \mid |\mathcal{N}_\epsilon(p)| < \text{min\_samples} \text{ and } p \notin \mathcal{N}_\epsilon(q) \text{ for any core point } q\}$$
Every point assigned label `-1` is flagged as an anomaly candidate.

---

## Experimental Results & Parameter Tuning

Experiments were conducted on the fixed 10,000-row subset (16 fraud cases, 0.16% base rate):

### Baseline vs. Optimal Model

| Metric | Baseline (`eps=8`, `min_samples=10`) | Optimal (`eps=8`, `min_samples=5`) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Clusters Found** | 1 | 1 | Same |
| **Anomalies Flagged** | 172 (1.72%) | **151 (1.51%)** | **-21 false alarms (-12.2%)** |
| **True Positives (Captured Fraud)** | 13 / 16 | **13 / 16** | Maintained high capture |
| **Recall** | **81.25%** | **81.25%** | Preserved |
| **Precision** | 7.56% | **8.61%** | **+13.9% relative gain** |
| **F1-Score** | 0.1383 | **0.1557** | **+12.6% relative gain** |

### Parameter Sensitivity Grid (Summary)

| `eps` | `min_samples` | Anomalies Flagged | Recall (%) | Precision (%) | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6.0 | 10 | 374 | 81.25% | 3.48% | 0.0667 |
| 7.0 | 10 | 256 | 81.25% | 5.08% | 0.0956 |
| 8.0 | 10 *(Baseline)* | 172 | 81.25% | 7.56% | 0.1383 |
| 9.0 | 10 | 134 | 50.00% | 5.97% | 0.1067 |
| **8.0** | **5 *(Optimal)*** | **151** | **81.25%** | **8.61%** | **0.1557** |
| 7.0 | 5 | 220 | 81.25% | 5.91% | 0.1102 |

### Multi-Cluster Experiment
Only a single configuration (`eps=6.0, min_samples=5`) split the data into 2 distinct clusters. However, this doubled the flagged anomalies to 330 while capturing the exact same 13 fraud cases, cutting Precision in half to 3.94%. This demonstrates that forcing higher cluster counts does not improve anomaly detection in financial spaces.

---

## Key Insights & Trade-Offs

1. **High Unsupervised Recall:** DBSCAN successfully isolated over **81%** of actual fraud cases without seeing a single training label, confirming that fraudulent patterns naturally occupy low-density regions.
2. **The Precision Dilemma:** Precision maxes out around 8.6%, meaning ~91% of flagged transactions are legitimate anomalies (e.g., unusually large purchases or atypical merchant activity). 
3. **Production Recommendation:** Unsupervised DBSCAN is best deployed as a **first-tier screening filter** that cuts 98.5% of normal traffic and routes the remaining 1.5% to downstream supervised models or fraud analyst review queues.

---

## Repository Structure

```text
dbscan-financial-anomaly-detection/
├── data/
│   └── sample_creditcard_10k.csv      # Ready-to-run 10,000-transaction sample
├── docs/
│   ├── Anomaly-Detection-in-Financial-Transactions-using-DBSCAN.pptx # Slide deck
│   └── DBSCAN_Anomaly_Detection_Report.docx                          # Complete academic report
├── src/
│   └── anomaly_detection_dbscan.py    # End-to-end Python pipeline & CLI
├── visualizations/
│   ├── k_distance_plot.png            # Sorted k-distance curve
│   ├── pca_clustering_anomaly_scatter.png # 2D PCA cluster projection
│   └── parameter_tuning_heatmap.png   # Parameter sensitivity heatmaps
├── .gitignore                         # Excludes large CSVs and environments
├── requirements.txt                   # Dependency definitions
└── README.md                          # Project documentation
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/shravan3161/dbscan-financial-anomaly-detection.git
cd dbscan-financial-anomaly-detection
```

### 2. Set Up Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Pipeline
The repository includes a ready-to-run 10,000-row sample (`data/sample_creditcard_10k.csv`):

```bash
# Run optimal configuration (eps=8.0, min_samples=5)
python src/anomaly_detection_dbscan.py

# Run full parameter sensitivity grid and regenerate heatmaps
python src/anomaly_detection_dbscan.py --grid_search
```

---

## References

1. Machine Learning Group - ULB and Worldline, *"Credit Card Fraud Detection,"* Kaggle Dataset ([Kaggle Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).
2. Dal Pozzolo et al., *"Calibrating probability with undersampling for unbalanced classification,"* IEEE SSCI 2015.
3. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996), *"A density-based algorithm for discovering clusters in large spatial databases with noise,"* KDD-96, pp. 226–231.
4. Scikit-learn developers, `sklearn.cluster.DBSCAN`, `sklearn.decomposition.PCA`, `sklearn.preprocessing.RobustScaler`.

---

## Author

* **Shravan Lingayat**
* [LinkedIn](https://www.linkedin.com) | [GitHub](https://github.com/shravan3161)
