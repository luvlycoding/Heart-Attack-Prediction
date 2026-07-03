# Up-to-Date Heart Attack Analysis and Prediction 🫀

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project to analyze clinical features and predict the risk of heart attacks. This repository contains the complete workflow—from Exploratory Data Analysis (EDA) and robust preprocessing to model evaluation and hyperparameter tuning of multiple classifiers.

---

## 📌 Project Overview
Heart disease remains one of the leading causes of mortality globally. Early diagnosis and preventative measures are critical. This project implements a classification pipeline using clinical datasets to assess whether a patient has a low or high chance of a heart attack.

The final model achieves **90.3% Test Accuracy** and **93% AUC-ROC** using a **Random Forest Classifier** tuned via GridSearchCV.

---

## 🗂️ Table of Contents
1. [Dataset Description](#-dataset-description)
2. [Workflow & Methodology](#-workflow--methodology)
   * [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
   * [2. Preprocessing & Feature Engineering](#2-preprocessing--feature-engineering)
   * [3. Machine Learning Modeling](#3-machine-learning-modeling)
3. [Model Evaluation & Results](#-model-evaluation--results)
4. [Installation & Requirements](#-installation--requirements)
5. [How to Run](#-how-to-run)
6. [Key Insights & Conclusion](#-key-insights--conclusion)

---

## 📊 Dataset Description
The model is trained on clinical data featuring various patient health metrics. The dataset includes 303 patient records and 14 clinical variables:

### Key Features:
* **`age`**: Age of the patient (years).
* **`sex`**: Sex of the patient (`1 = male`; `0 = female`).
* **`cp`**: Chest pain type:
  * `0`: Asymptomatic
  * `1`: Typical angina
  * `2`: Atypical angina
  * `3`: Non-anginal pain
* **`trtbps`**: Resting blood pressure (in mm Hg on admission to the hospital).
* **`chol`**: Serum cholesterol in mg/dl. *(Dropped during feature selection)*
* **`fbs`**: Fasting blood sugar > 120 mg/dl (`1 = true`; `0 = false`). *(Dropped during feature selection)*
* **`rest_ecg`**: Resting electrocardiographic results:
  * `0`: Hypertrophy
  * `1`: Normal
  * `2`: ST-T wave abnormality
  *(Dropped during feature selection)*
* **`thalach`**: Maximum heart rate achieved.
* **`exang`**: Exercise-induced angina (`1 = yes`; `0 = no`).
* **`oldpeak`**: ST depression induced by exercise relative to rest.
* **`slope`**: The slope of the peak exercise ST segment (`0 = downsloping`; `1 = flat`; `2 = upsloping`).
* **`ca`**: Number of major vessels (0-3) colored by fluoroscopy.
* **`thal`**: Thallium stress test result (`1 = fixed defect`; `2 = normal`; `3 = reversible defect`).
* **`target`**: Predicted heart attack risk (`0 = less chance of heart attack`; `1 = more chance of heart attack`).

---

## 🛠️ Workflow & Methodology

### 1. Exploratory Data Analysis (EDA)
* **Univariate Analysis**: Explored numeric features using distributions (`sns.distplot`) and categorical features using Pie Charts.
* **Bivariate Analysis**:
  * Plotted `FacetGrid` KDE plots of numeric features against the target variable to uncover distributions.
  * Used `CountPlot` for categorical features vs. target.
  * Generated `pairplot` and `heatmap` to identify feature-to-feature and feature-to-target correlations.

### 2. Preprocessing & Feature Engineering
* **Feature Selection**: Dropped columns with extremely low correlation to the target: `chol`, `fbs`, and `rest_ecg`.
* **Missing Value Imputation**: Handled invalid/missing data in the Thallium stress test (`thal`) column by imputing values of `0` with the median score of `2`.
* **Outlier Handling**:
  * **Winsorization**: Capped outliers in resting blood pressure (`trtbps`) and ST depression (`oldpeak`) to mitigate the effect of extreme values.
  * **Data Cleaning**: Removed a single, highly anomalous outlier row (index `272`) for maximum heart rate (`thalach`).
* **Feature Transformations**: Applied a square-root transformation (`np.sqrt`) to the winsorized `oldpeak` column to correct severe right skewness.
* **Categorical Encoding**: Converted multi-class variables to binary dummy variables using One-Hot Encoding (`pd.get_dummies`), applying `drop_first=True` to avoid multicollinearity.
* **Feature Scaling**: Scaled all numerical inputs (`age`, `thalach`, `trtbps_winsorize`, `oldpeak_winsorize_sqrt`) using `RobustScaler` to ensure they are robust to remaining outliers.
* **Data Splitting**: Split data into a 90% Training Set and a 10% Test Set (`random_state=3`).

### 3. Machine Learning Modeling
Evaluated four classification models using metrics like **Accuracy**, **Area Under Curve (AUC-ROC)**, and **10-Fold Cross-Validation (CV) Accuracy**:
* **Logistic Regression** (optimized solver/penalty via `GridSearchCV`)
* **Decision Tree Classifier**
* **Support Vector Machine (SVC)**
* **Random Forest Classifier** (tuned `n_estimators`, `criterion`, and `max_features` via `GridSearchCV`)

---

## 🏆 Model Evaluation & Results

Below is the comparison of model performance metrics on the test split:

| Classifier Model | Test Accuracy | AUC-ROC | Cross-Validation Score (Mean) |
| :--- | :---: | :---: | :---: |
| **Random Forest Classifier (Tuned)** | **90.3%** | **93.0%** | **83.3%** |
| **Logistic Regression (Tuned)** | **87.1%** | **88.0%** | **87.0%** |
| **Support Vector Classifier (SVC)** | **83.9%** | **89.0%** | **83.3%** |
| **Decision Tree Classifier** | **83.9%** | **85.0%** | **60.0%** |

*The tuned Random Forest Classifier outperformed other architectures in terms of test accuracy (90.3%) and classification probability stability (93.0% AUC-ROC).*

---

## 💻 Installation & Requirements

Ensure you have Python 3.11+ installed. Install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn missingno scipy
```

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/heart-attack-analysis-prediction.git
   cd heart-attack-analysis-prediction
   ```
2. Place the dataset `heart.csv` in the appropriate input directory.
3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook up-to-date-heart-attack-analysis-and-prediction(7).ipynb
   ```

---

## 🧠 Key Insights & Conclusion
1. **Age vs. Risk Paradox**: The analysis revealed a negative correlation between age and heart attack target score (older patients in this specific clinical cohort actually presented a slightly lower probability score), indicating selection bias or specific subgroup characteristics in the study.
2. **Key Physical Indicators**: A higher maximum heart rate achieved (`thalach`) and typical/atypical chest pain types (`cp`) are strong positive indicators of heart attack risk.
3. **Model Selection**: Tuning ensemble models like **Random Forest** offers robust performance improvements on clinical tabular datasets over simpler linear boundaries.
