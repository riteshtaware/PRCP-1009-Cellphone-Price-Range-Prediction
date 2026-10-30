# PRCP-1009-Cellphone-Price-Range-Prediction
# 📱 Cellphone Price Range Prediction

A machine learning classification project that predicts the **price range** of a mobile phone (0 = low, 1 = medium, 2 = high, 3 = very high) from its technical specifications.

## 📌 Business Problem

Bob has started a new mobile company and wants to compete with the large brands, but he doesn't know how to price the phones his company builds. This project analyses mobile-phone sales data to:

1. Identify which specifications drive price.
2. Build and compare multiple classification models.
3. Recommend the best model for production.
4. Show how the model can help Bob price products and expand his business.

> **Note:** The model predicts a price *range* (0 to 3), not the exact selling price. It is based on historical patterns in the dataset and does not account for brand value, marketing, or market timing.

## 📂 Repository Structure

```
├── PRCP-1009-CellphonePrice.ipynb   # Full analysis notebook
├── datasets_11167_15520_train.csv   # Dataset
└── README.md
```

## 📊 Dataset

| Property | Detail |
|---|---|
| Rows | 2,000 phones |
| Features | 20 specifications + target (`price_range`) |
| Missing values | None |
| Duplicates | None |
| Target balance | Perfectly balanced (500 phones per class) |

**Feature groups**

- **Performance:** `ram`, `battery_power`, `clock_speed`, `n_cores`, `int_memory`
- **Display:** `px_height`, `px_width`, `sc_h`, `sc_w`
- **Camera:** `fc` (front camera MP), `pc` (primary camera MP)
- **Physical:** `mobile_wt`, `m_dep`
- **Connectivity and extras:** `blue`, `dual_sim`, `three_g`, `four_g`, `wifi`, `touch_screen`, `talk_time`

## 🧹 Data Cleaning

- `sc_w` had **180** impossible zeros and `px_height` had **2**. These were replaced with the median of the non-zero values, since the zeros were spread evenly across price classes and looked like data-entry gaps.
- Zeros in `fc` and `pc` were **kept**, because a phone can legitimately have no camera.
- A few high `fc` values were kept as genuine specifications.

## 🔍 Key EDA Findings

- **RAM is by far the strongest driver of price** (correlation with price range of about 0.92).
- Battery power and screen resolution (`px_width`, `px_height`) have weak positive relationships with price.
- Bluetooth, Wi-Fi, 3G/4G, dual SIM, touch screen, clock speed, core count, weight and talk time show almost no relationship with price in this data.
- No serious multicollinearity (highest feature-to-feature correlation is about 0.65).

## 🛠️ Feature Engineering

Two features were created, giving 22 features in total:

- `px_total` = `px_height` × `px_width`
- `screen_area` = `sc_h` × `sc_w`

## 🤖 Modelling Approach

- Stratified **80/20 train-test split** (1,600 train / 400 test, `random_state=42`).
- **5-fold stratified cross-validation** for model comparison.
- Scale-sensitive models were wrapped in pipelines with `StandardScaler`.
- **GridSearchCV** was used to tune Logistic Regression, SVM, Gradient Boosting and Random Forest.
- Metrics: accuracy, macro precision, recall, F1, ROC-AUC and CV accuracy.

**Models evaluated:** Logistic Regression, KNN, SVM (RBF), Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost, plus tuned versions of four of them.

## 🏆 Results

| Model | Train Acc | Test Acc | CV Acc | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Tuned Logistic Regression** | 0.989 | **0.980** | **0.963** | 0.980 | 0.999 |
| Tuned SVM (linear) | 0.984 | 0.970 | 0.960 | 0.970 | 0.999 |
| Logistic Regression | 0.976 | 0.962 | 0.947 | 0.962 | 0.999 |
| XGBoost | 1.000 | 0.930 | 0.908 | 0.930 | 0.994 |
| Tuned Gradient Boosting | 1.000 | 0.930 | 0.900 | 0.930 | 0.993 |
| Gradient Boosting | 1.000 | 0.920 | 0.894 | 0.920 | 0.991 |
| Tuned Random Forest | 1.000 | 0.885 | 0.880 | 0.884 | 0.982 |
| Random Forest | 1.000 | 0.888 | 0.866 | 0.887 | 0.981 |
| SVM (RBF) | 0.982 | 0.860 | 0.853 | 0.860 | 0.982 |
| Decision Tree | 1.000 | 0.855 | 0.837 | 0.854 | 0.903 |
| Naive Bayes | 0.809 | 0.815 | 0.804 | 0.816 | 0.955 |
| KNN | 0.696 | 0.510 | 0.486 | 0.516 | 0.748 |

**Observations**

- **Tuned Logistic Regression** is the best model, with 98.0% test accuracy and a macro F1 of 0.98.
- Tree-based models reach 100% training accuracy but lower test accuracy, which indicates **overfitting**.
- KNN performs worst because distance is dominated by uninformative features.
- Linear models win because the price class rises almost linearly with RAM, battery and resolution.
- **Noise-feature experiment:** using only the top 5 features (`ram`, `battery_power`, `px_height`, `px_width`, `px_total`) improved Random Forest from 88.8% to 95.2%, while Logistic Regression stayed about the same (96.2% to 96.8%).
- All errors of the best model were between **neighbouring** price classes.

## ✅ Recommendation

Use **Tuned Logistic Regression** in production because it has:

- the highest cross-validated accuracy,
- a small train-test gap (98.9% vs 98.0%),
- very fast training and prediction,
- interpretable coefficients showing how each specification pushes the price range up or down, which is what Bob needs to understand his pricing.

## 💼 Business Takeaways

- **RAM is the main pricing lever**, with battery power and display resolution as secondary factors.
- Features such as Bluetooth, Wi-Fi, touch screen or number of cores have little effect on price tier in this data, so they are not where pricing power comes from.

## ⚠️ Limitations

- The dataset is clean and the classes are very linearly separable. Real market data (brand, launch date, discounts) would be messier.
- The test set has only 400 phones, so differences of about 1% between the top models are within noise.
- The model predicts a price tier, not an exact price.

## 🧰 Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · Jupyter Notebook

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```
3. Make sure `datasets_11167_15520_train.csv` is in the same folder as the notebook.
4. Launch Jupyter and run the notebook:
   ```bash
   jupyter notebook PRCP-1009-CellphonePrice.ipynb
   ```

## 👤 Author

**<Your Name>**
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
