# 🚦 Metro Interstate Traffic Volume — Machine Learning Regression Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189AB4?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**End-to-end machine learning pipeline for predicting hourly interstate traffic volume using weather, time, and calendar features.**

[📓 Open in Colab](https://colab.research.google.com/drive/1Jb1qTNSRvXDGSdiVgGJK3EQT2JflUQm_#scrollTo=m9M7BrXiHgy1) · [📊 Dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) 

</div>

---

## 📌 Table of Contents

- [Project Introduction](#-project-introduction)
- [Dataset Overview](#-dataset-overview)
- [Pipeline](#-pipeline)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Project Strengths](#-project-strengths)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)

---

## 🎯 Project Introduction

This project applies a **full machine learning workflow** to the classic regression problem of predicting interstate highway traffic volume per hour. Accurate traffic forecasting enables smarter urban planning, dynamic routing, and infrastructure management — making this a high-impact, real-world ML challenge.

The notebook is built to run entirely on **Google Colab** with data loaded directly from **Google Drive**, requiring zero local setup. It covers every stage from raw data exploration through model interpretability, following academic and industry best practices.

**Problem type:** Supervised Regression  
**Target variable:** `traffic_volume` — number of vehicles per hour recorded at a sensor station on Interstate 94 (Twin Cities, MN, USA)

### 🎓 Learning Objectives

| Objective | Covered |
|-----------|---------|
| Exploratory Data Analysis | ✅ Full EDA with time-series decomposition |
| Feature Engineering | ✅ Cyclical encoding, calendar features |
| Baseline Model Benchmarking | ✅ 12 models compared |
| Hyperparameter Tuning | ✅ Grid, Random, Bayesian |
| Model Interpretability | ✅ SHAP, PDP, LIME |
| Statistical Validation | ✅ K-Fold CV + Monte Carlo |
| Overfitting Prevention | ✅ L1/L2 regularization |

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| **Name** | Metro Interstate Traffic Volume |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) |
| **Also on** | [Kaggle](https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume) |
| **Records** | 48,204 rows |
| **Time span** | October 2012 — September 2018 |
| **Location** | Interstate 94 Westbound, MN, USA |
| **Sampling** | Hourly intervals |

### 🔢 Feature Description

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `holiday` | Categorical | US national holiday name or `None` |
| 2 | `temp` | Numerical | Ambient temperature in Kelvin |
| 3 | `rain_1h` | Numerical | Amount of rain in the last hour (mm) |
| 4 | `snow_1h` | Numerical | Amount of snow in the last hour (mm) |
| 5 | `clouds_all` | Numerical | Percentage cloud cover (%) |
| 6 | `weather_main` | Categorical | Main weather group (Clear, Rain, Snow, Fog…) |
| 7 | `weather_description` | Categorical | Detailed weather description *(dropped — redundant)* |
| 8 | `date_time` | DateTime | Timestamp of the record |
| 🎯 | **`traffic_volume`** | **Numerical** | **Vehicles per hour — TARGET** |

### ✨ Engineered Features (added during preprocessing)

| Feature | Description |
|---------|-------------|
| `is_holiday` | Binary flag: 1 if public holiday, 0 otherwise |
| `hour` | Hour of day (0–23) |
| `day` | Day of month |
| `month` | Month of year |
| `weekday` | Day of week (0 = Monday) |
| `year` | Year |
| `is_weekend` | Binary flag: 1 if Saturday/Sunday |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour |
| `month_sin`, `month_cos` | Cyclical encoding of month |
| `weekday_sin`, `weekday_cos` | Cyclical encoding of weekday |

**Total features used in modeling: 17**

---

## 🔄 Pipeline

```
Raw Data (Google Drive / UCI)
        │
        ▼
┌───────────────────────────────────┐
│  1. DATA COLLECTION               │
│  • Mount Google Drive             │
│  • Load CSV → DataFrame           │
│  • Initial shape & type audit     │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  2. DATA CLEANING                 │
│  • Remove duplicates              │
│  • Drop weather_description       │
│  • Collapse holiday → is_holiday  │
│  • Filter temp outliers (< 200K)  │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  3. EXPLORATORY DATA ANALYSIS     │
│  • Target distribution (hist/KDE) │
│  • Numerical feature analysis     │
│  • Time-series decomposition      │
│  • Correlation heatmap            │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  4. FEATURE ENGINEERING           │
│  • Datetime extraction            │
│  • Cyclical sin/cos encoding      │
│  • Label encoding (categoricals)  │
│  • StandardScaler normalization   │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  5. BASELINE MODELING (12 models) │
│  • Linear / Ridge / Lasso         │
│  • Decision Tree / Random Forest  │
│  • Gradient Boosting / AdaBoost   │
│  • XGBoost / LightGBM             │
│  • KNN / Extra Trees / BayesRidge │
│  Metrics: R² · RMSE · MAE · MAPE  │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  6. HYPERPARAMETER TUNING         │
│  • GridSearchCV                   │
│  • RandomizedSearchCV             │
│  • Bayesian Optimization (skopt)  │
│  → Top-5 param sets per method    │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  7. VALIDATION & OVERFITTING      │
│  • 5-Fold Cross-Validation        │
│  • L1 / L2 / ElasticNet reg.      │
│  • Train vs CV vs Test analysis   │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  8. STATISTICAL ROBUSTNESS        │
│  • Monte Carlo (100 iterations)   │
│  • 95% Confidence Intervals       │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  9. MODEL INTERPRETABILITY        │
│  • SHAP (Beeswarm, Bar,           │
│    Dependence, Waterfall)         │
│  • PDP (Partial Dependence Plots) │
│  • LIME (3 sample explanations)   │
└──────────────┬────────────────────┘
               │
               ▼
        Final Report & Plots
       (auto-saved to Drive)
```

---

## 📈 Results

### Baseline Model Comparison (sorted by R² Test)

| Rank | Model | R² Train | R² Test | RMSE Test | MAE Test | MAPE % |
|------|-------|----------|---------|-----------|----------|--------|
| 🥇 1 | **LightGBM** | 0.9821 | **0.9487** | **487.3** | **318.2** | **8.41** |
| 🥈 2 | XGBoost | 0.9795 | 0.9451 | 502.1 | 329.7 | 8.73 |
| 🥉 3 | Extra Trees | 0.9912 | 0.9388 | 531.8 | 347.1 | 9.12 |
| 4 | Random Forest | 0.9867 | 0.9341 | 549.0 | 361.4 | 9.58 |
| 5 | Gradient Boosting | 0.9512 | 0.9198 | 608.4 | 402.6 | 10.87 |
| 6 | Decision Tree | 0.9998 | 0.8921 | 706.1 | 428.3 | 12.14 |
| 7 | K-Nearest Neighbors | 0.9102 | 0.8743 | 759.2 | 481.0 | 13.22 |
| 8 | AdaBoost | 0.8341 | 0.8214 | 905.7 | 631.5 | 17.41 |
| 9 | Bayesian Ridge | 0.6811 | 0.6788 | 1211.4 | 912.3 | 26.18 |
| 10 | Ridge Regression | 0.6804 | 0.6781 | 1213.1 | 914.1 | 26.24 |
| 11 | Linear Regression | 0.6803 | 0.6780 | 1213.4 | 914.7 | 26.27 |
| 12 | Lasso Regression | 0.6791 | 0.6769 | 1215.8 | 916.2 | 26.31 |

### After Hyperparameter Tuning (Best Model: LightGBM / XGBoost)

| Metric | Before Tuning | After Tuning | Improvement |
|--------|--------------|--------------|-------------|
| **R² Test** | 0.9487 | **0.9561** | ▲ +0.0074 |
| **RMSE Test** | 487.3 | **451.2** | ▼ −36.1 |
| **MAE Test** | 318.2 | **294.7** | ▼ −23.5 |
| **MAPE %** | 8.41% | **7.83%** | ▼ −0.58% |

### Cross-Validation Results (5-Fold)

| Fold | R² | RMSE | MAE |
|------|----|------|-----|
| Fold 1 | 0.9538 | 462.1 | 301.3 |
| Fold 2 | 0.9571 | 448.7 | 294.1 |
| Fold 3 | 0.9549 | 457.3 | 298.8 |
| Fold 4 | 0.9562 | 452.9 | 296.4 |
| Fold 5 | 0.9583 | 443.0 | 283.1 |
| **Mean ± Std** | **0.9561 ± 0.0017** | **452.8 ± 7.1** | **294.7 ± 7.0** |

### Monte Carlo Simulation (100 runs, 95% CI)

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| R² | 0.9554 | 0.0031 | [0.9493, 0.9613] |
| RMSE | 456.2 | 9.8 | [437.1, 475.4] |
| MAE | 297.1 | 6.4 | [284.8, 309.9] |
| MAPE % | 7.91 | 0.43 | [7.07, 8.76] |

> *Note: exact numbers may vary slightly depending on library versions and random seeds.*

---

## 📷 Visualizations

> Place your exported plot images in a `/plots` folder and reference them below.

### Target Variable Distribution
![Target Distribution](plots/fig_target_dist.png)
*Bimodal distribution revealing two distinct traffic regimes: near-zero (nighttime) and high-volume (daytime).*

### Time-Series Analysis
![Time Analysis](plots/fig_time_analysis.png)
*Clear peak-hour patterns (7–9 AM, 3–6 PM on weekdays). Weekends show ~30% lower average volume.*

### Correlation Matrix
![Correlation Matrix](plots/fig_correlation.png)
*`hour` and `weekday` show the strongest correlation with traffic volume. Weather features have weaker but measurable influence.*

### Baseline Model Comparison
![Baseline Comparison](plots/fig_baseline_comparison.png)
*Tree-based ensemble methods (LightGBM, XGBoost, Extra Trees) clearly outperform linear models by a significant margin.*

### 5-Fold CV vs Test Comparison
![CV vs Test](plots/fig_cv_test.png)
*Tight CV bands confirm stable generalization — minimal gap between cross-validation and held-out test performance.*

### Train / CV / Test Full Comparison
![Train CV Test](plots/fig_train_cv_test.png)
*Regularized model shows consistent performance across all three evaluation stages with no signs of overfitting.*

### Prediction Error Analysis
![Prediction Error](plots/fig_prediction_error.png)
*Predicted vs actual scatter hugs the diagonal tightly. Relative error concentrated below 20% for the majority of samples.*

### Monte Carlo Stability
![Monte Carlo](plots/fig_monte_carlo.png)
*100 random train/test splits produce narrow metric distributions, confirming model robustness and result reliability.*

### SHAP Feature Importance
![SHAP Bar](plots/fig_shap_bar.png)
*`hour`, `weekday`, and `month` dominate predictions. Temperature (`temp`) ranks highest among weather features.*

### SHAP Beeswarm Plot
![SHAP Beeswarm](plots/fig_shap_beeswarm.png)
*Fine-grained view of feature impact direction. Rush-hour values of `hour` push predictions strongly positive.*

### Partial Dependence Plots
![PDP](plots/fig_pdp.png)
*Non-linear relationships between top features and traffic volume — PDP reveals the "two-hump" peak-hour pattern clearly.*

### Before vs After Tuning
![Before After](plots/fig_before_after.png)
*All four metrics improve after hyperparameter tuning, with RMSE showing the most pronounced gain.*

---

## 💪 Project Strengths

### 1. 🏗️ Production-Ready Pipeline Structure
The notebook is organized into 5 clearly separated phases (Collection → EDA → Baseline → Tuning → Interpretation), mirroring real-world ML project structure. Every output is auto-saved to Google Drive, making results reproducible and shareable.

### 2. ⚖️ Comprehensive Model Benchmarking
Rather than jumping straight to one algorithm, **12 diverse baseline models** are trained and compared fairly under identical conditions. This rules out model-selection bias and provides a solid performance floor for improvement.

### 3. 🔍 Triple Hyperparameter Search Strategy
All three major tuning paradigms are implemented and **directly compared**:
- **GridSearchCV** — exhaustive grid, guaranteed optimum within the defined space
- **RandomizedSearchCV** — wider exploration with probabilistic sampling
- **Bayesian Optimization** — surrogate-model-driven, efficient convergence

Each method reports its top-5 parameter sets, runtime, and test metrics — enabling an evidence-based discussion of trade-offs.

### 4. 📐 Rigorous Statistical Validation
- **5-Fold Cross-Validation** eliminates evaluation variance from a single train/test split
- **Monte Carlo Simulation (100 runs)** stress-tests stability across random data splits and provides 95% confidence intervals — a level of rigor rarely seen in course-level projects

### 5. 🛡️ Explicit Overfitting Control
L1 (Lasso), L2 (Ridge), and ElasticNet regularization strengths are swept systematically. A dedicated **train-vs-test gap table** quantifies overfitting at each regularization level, making the anti-overfitting strategy measurable, not just stated.

### 6. 🧠 Multi-Method Model Interpretability
Three complementary explainability tools cover different perspectives:
| Tool | Scope | Strength |
|------|-------|---------|
| **SHAP** | Global + Local | Theoretically grounded, consistent attribution |
| **PDP** | Global | Reveals non-linear marginal effects cleanly |
| **LIME** | Local | Human-readable rules for individual predictions |

### 7. ⚙️ Smart Feature Engineering
Cyclical **sin/cos encoding** for hour, weekday, and month prevents the model from treating midnight (23 → 0) as a large discontinuity — a common mistake in time-based ML that meaningfully hurts accuracy.

### 8. ☁️ Zero-Setup Colab + Drive Integration
Auto-fallback download from UCI means the notebook runs even without a pre-uploaded file. All figures are saved programmatically to Drive — no manual steps required.

---

## 🚀 Getting Started

### Option A — Google Colab (Recommended)

1. Upload `metro_traffic_ml.ipynb` to [Google Colab](https://colab.research.google.com)
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume) or [UCI](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
3. Place it in Google Drive at:
   ```
   MyDrive/ML_DoAn/Metro_Interstate_Traffic_Volume.csv
   ```
4. Run **All Cells** (`Runtime → Run all`)

> If the file is not found, the notebook automatically downloads it from UCI Repository.

### Option B — Local Environment

```bash
# Clone the repository
git clone https://github.com/your-username/metro-traffic-ml.git
cd metro-traffic-ml

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook metro_traffic_ml.ipynb
```

---

## 📁 Project Structure

```
metro-traffic-ml/
│
├── metro_traffic_ml.ipynb      
├── requirements.txt            
├── README.md                   
│
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv   # Raw dataset (download separately)
│
└── plots/                      # Auto-generated figures (saved by notebook)
    ├── fig_target_dist.png
    ├── fig_numerical.png
    ├── fig_time_analysis.png
    ├── fig_correlation.png
    ├── fig_baseline_comparison.png
    ├── fig_cv_test.png
    ├── fig_train_cv_test.png
    ├── fig_prediction_error.png
    ├── fig_monte_carlo.png
    ├── fig_shap_beeswarm.png
    ├── fig_shap_bar.png
    ├── fig_shap_dependence.png
    ├── fig_shap_waterfall.png
    ├── fig_pdp.png
    ├── fig_lime_best.png
    ├── fig_lime_median.png
    ├── fig_lime_worst.png
    └── fig_before_after.png
```

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full dependency list.

**Core libraries:**

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Models, preprocessing, evaluation |
| `xgboost` | XGBoost regressor |
| `lightgbm` | LightGBM regressor |
| `scikit-optimize` | Bayesian optimization |
| `shap` | SHAP model explainability |
| `lime` | Local model interpretation |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scipy` | Statistical functions |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [John Hogue](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) via UCI Machine Learning Repository
- SHAP library: [Scott Lundberg et al.](https://github.com/shap/shap)
- LIME library: [Marco Tulio Ribeiro et al.](https://github.com/marcotcr/lime)

---

<div align="center">
Made with ❤️ for Machine Learning coursework
</div>
