# 🛡️ Credit Card Fraud Detection

This project tackles the critical task of detecting fraudulent credit card transactions using machine learning. With only **0.17%** of transactions being frauds, handling **imbalanced data** and building a robust, scalable model is the key challenge.

---

## 📌 Problem Statement

Credit card fraud poses significant financial threats. Our goal is to accurately classify fraudulent transactions.

---

## 📊 Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Samples:** 284,807 transactions
- **Fraud Cases:** ~492 (0.17%)
- **Features:** 30 (V1 to V28 are PCA components, plus `Time` and `Amount`)
- **Target:** `Class` → 0 = Legit, 1 = Fraud

---

## 🧽 Data Cleaning

✅ Removed the following:
- `Time` column (not meaningful)
- **Duplicate rows** to reduce bias

---

## 📊 Exploratory Data Analysis (EDA)

The dataset underwent a detailed exploratory analysis to uncover patterns, identify outliers, and guide preprocessing:

- 📉 **Class Distribution (Count Plot):**  
  A bar plot revealed an **extremely imbalanced dataset**, with fraudulent transactions accounting for less than **0.2%** of the total data.

- 📊 **Feature Distributions (Histograms):**  
  Histograms were plotted for all PCA-transformed features (`V1`–`V28`) to understand their spread and detect any unusual distributions.

- 💰 **Transaction Amount Analysis:**  
  - The `Amount` feature exhibited **strong right skewness**.  
  - To correct this, a **logarithmic transformation** was applied, significantly normalizing its distribution.

- 📦 **Box Plots vs Class:**  
  - Constructed box plots for both raw `Amount` and `Log Amount` grouped by the `Class` label.  
  - This helped visualize how transaction amounts differ between **legitimate** and **fraudulent** transactions.

- 🔥 **Correlation Heatmap:**  
  - A correlation matrix was generated to examine relationships between all features (`V1`–`V28`) and the target `Class`.  
  - This helped identify **features with higher predictive potential** for fraud detection.

---

## ⚙️ Preprocessing

- Applied **log transformation** on `Amount` to address skewness
- Applied **StandardScaler** on the log-transformed `Amount`
- Final Preprocessing Steps:
  - ➗ Train-Test Split (80/20 with stratify)
  - ⚖️ **SMOTE Oversampling** on training set to balance classes

---

## 🤖 Model Training & Evaluation

Trained 3 models for comparison:
--Logistic Regression 
--Decision Tree Classifier
--Random Forest Classifier

- Evaluated using:
  - Accuracy
  - Classification Report
  - Confusion Matrix
  - ROC-AUC Score

---

## 🧠 Hyperparameter Tuning (Random Forest)

Used **GridSearchCV** with 3-fold CV to tune:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `bootstrap`

Achieved:
- ⚡ Improved recall and precision
- : F1 Score **~0.92**

---

## Final Pipeline

```bash
1. Load & clean and inspect data
2. Perform EDA & visualize patterns
3. Log-transform + standardize `Amount`
4. Train-test split
5. Apply SMOTE to balance classes
6. Train multiple models (Logistic, Tree, RF)
7. Tune Random Forest using GridSearchCV
8. Evaluate on test set
