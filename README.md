# 🌍 Global health Data Analysis and Prediction

This project explores and analyzes the **Life Expectancy Dataset** to understand the key factors influencing life expectancy across the globe. It involves data preprocessing, exploratory data analysis (EDA), correlation studies, visualization, and machine learning modeling for regression and classification tasks.

---

## 📁 Dataset

**Source**: [Kaggle Life Expectancy Data](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

**Attributes**:
- Country, Year, Status (Developed/Developing)
- Life Expectancy, Adult Mortality, BMI, GDP, Schooling, Immunization stats (Polio, Diphtheria), Alcohol consumption, and more.

---

## 🔍 Objectives

- Handle missing values and clean the dataset.
- Perform EDA to extract trends and insights.
- Visualize key factors affecting life expectancy.
- Apply machine learning models to predict life expectancy (regression).
- Classify countries based on life expectancy (binary classification).

---

## 🧪 Libraries Used

- `pandas`, `numpy` – Data manipulation
- `matplotlib`, `seaborn` – Visualization
- `dtale` – Interactive data exploration
- `scikit-learn` – ML modeling and preprocessing

---

## 📊 Exploratory Data Analysis

Visualizations and key insights:

1. **Distribution of Life Expectancy**: Normal distribution centered around ~70 years.
2. **Developed vs Developing**: Developed countries show significantly higher life expectancy.
3. **Top 10 Countries**: Bar charts of countries with highest/lowest average life expectancy.
4. **Correlation Heatmap**: Positive correlation with schooling, BMI; negative with adult mortality.
5. **Trends Over Time**: Life expectancy generally increases with time.
6. **Scatterplots**: Explored relationships between life expectancy and Schooling, GDP, BMI, Immunization (Polio).

---

## ⚙️ Data Preprocessing

- Handled missing values using column means and group-wise means.
- Normalized features using `StandardScaler`.

---

## 🧠 Machine Learning

### Regression Models

| Model               | R² Score | Mean Squared Error | Mean Absolute Error |
|--------------------|----------|--------------------|---------------------|
| Linear Regression  | ~0.88    | ~5.1               | ~1.8                |
| Support Vector Regressor (SVR) | ~0.86    | ~6.0               | ~1.9                |

🏆 **Best Model**: Linear Regression

### Classification Model

- **Target**: Binary classification – High Life Expectancy (>70 years) vs Low (≤70 years)
- **Model**: K-Nearest Neighbors (KNN)
- **Accuracy**: ~0.89
- **Metrics**: Confusion Matrix, Precision, Recall, F1-score

---

## 🧾 How to Run

1. Clone this repository
2. Install the dependencies:
   ```bash
   pip install pandas seaborn matplotlib dtale scikit-learn
