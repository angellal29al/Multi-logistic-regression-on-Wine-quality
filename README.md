# 🍷 Wine Quality Prediction Using Multiclass Logistic Regression

## 📌 Objective
The aim of this project is to apply a **multiclass classification algorithm** to a business analytics problem within the **Finance or Consumer Goods domain**. Specifically, the model predicts the **quality of red wine** based on physicochemical test results using **multiclass logistic regression**.

---

## 📂 Dataset Description

- **Source**: Red Vinho Verde wine samples from the north of Portugal.
- **Goal**: Predict the **wine quality**, a categorical variable, using chemical properties such as:
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol

---

## 🔍 Exploratory Data Analysis (EDA)

Several visualizations were created to understand the dataset:
- Distribution of wine quality classes
- Pair plots to examine relationships between key variables
- Correlation heatmap of physicochemical properties

### Key Observations
- The dataset is **imbalanced**, with majority classes being **quality 5 and 6**

---

## 🧪 Preprocessing

To address class imbalance:
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to **oversample minority classes** in the training dataset
- Ensured **equal representation** of all quality classes (3 to 8) before model training

---

## 🤖 Model: Multiclass Logistic Regression

Used `LogisticRegression(multi_class='multinomial', solver='lbfgs')` from `sklearn`.

### Model Performance

- **Accuracy**: 63%
- **Precision (weighted avg)**: 58%
- **Recall (weighted avg)**: 63%

### Confusion Matrix Insights

| Class | Correct Predictions | Wrong Predictions |
|-------|---------------------|-------------------|
| 3     | 318                 | 2                 |
| 4     | 307                 | 11                |
| 5     | High Precision (0.67) | -               |
| 6     | High Precision (0.61) | -               |

---

## 📊 Evaluation Metrics

- **Confusion Matrix**: Interpreted to analyze True Positives, True Negatives, False Positives, and False Negatives for each class
- **Weighted Precision & Recall**: Chosen over macro/micro averages due to SMOTE-based oversampling

---

## ✅ Conclusion

- This project demonstrates the successful application of **multiclass logistic regression** to predict **wine quality** based on measurable chemical properties.
- Class imbalance was addressed using **SMOTE**, leading to more balanced predictions across all wine quality classes.
- While the model achieves **63% accuracy**, there's room for improvement using more complex models (e.g., Random Forest, Gradient Boosting)

---
