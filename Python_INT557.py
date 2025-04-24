# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 02:08:56 2025

@author: shiva
"""
import pandas as pd 
import matplotlib.pyplot as plt
import dtale as dt
import seaborn as sns

# load dataset
df=pd.read_csv("C:\\Users\\shiva\\OneDrive\\Desktop\\Life Expectancy Data.csv")
data = dt.show(df)
data.open_browser()

# about data
df.info()
df.describe()

print("Initial shape:", df.shape)
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

df.columns = df.columns.str.replace(' ', '')
df.fillna({
    'Life_expectancy':df['Life_expectancy'].mean(),
    'AdultMortality':df['AdultMortality'].mean(),
    'BMI':df['BMI'].mean(),
    'Polio':df['Polio'].mean(),
    'Diphtheria':df['Diphtheria'].mean(),
    'thinness1-19years':df['thinness1-19years'].mean(),
    'thinness5-9years':df['thinness5-9years'].mean()}, inplace=True)

cols_to_fill_with_mean = [
    'Alcohol', 'Hepatitis_B', 'Total_expenditure', 
    'GDP', 'Population', 'Income_composition_of_resources', 'Schooling'
]

for col in cols_to_fill_with_mean:
    df[col] = df.groupby('Country')[col].transform(lambda x: x.fillna(x.mean()))

#Distribution of Life Expectancy
sns.histplot(df['Life_expectancy'], kde=True, color='red')
plt.title('Distribution of Life Expectancy')
plt.xlabel('Life Expectancy')
plt.show()#life expectancy is approximately normally distributed with a peak around 70 years.

# Life Expectancy by Status (Developed vs Developing)

sns.boxplot(x='Status', y='Life_expectancy', data=df)
plt.title('Life Expectancy by Development Status')
plt.show()#Developed countries have significantly higher life expectancy than developing ones.

top_countries = df.groupby('Country')['Life_expectancy'].mean().sort_values(ascending=True).head(10)
top_countries.plot(kind='bar', color='seagreen')
plt.title('Top 10 Countries with Lowest Average Life Expectancy')
plt.ylabel('Life Expectancy')
plt.show()
# correlation 
cols = [
    'Life_expectancy', 'Alcohol','AdultMortality', 'Population',
    'Income_composition_of_resources', 'Schooling','BMI'
]
corr = df[cols].corr()  # Calculate correlation matrix
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
#4. Life Expectancy vs Adult Mortality4. Life Expectancy vs Adult Mortality
sns.scatterplot(x='Life_expectancy', y='AdultMortality', hue='Status', data=df)
plt.title('Life Expectancy vs Adult Mortality')
plt.show() #Insight: Higher adult mortality is strongly associated with lower life expectancy.

#5  Trend of Life Expectancy Over the Years
sns.lineplot(x='Year', y='Life_expectancy', data=df)
plt.title('Trend of Life Expectancy Over the Years')
plt.show() #Insight: Life expectancy has generally increased over time globally.
 
#6  Top 10 Countries with Highest Life Expectancy
top_countries = df.groupby('Country')['Life_expectancy'].mean().sort_values(ascending=False).head(10)
top_countries.plot(kind='bar', color='seagreen')
plt.title('Top 10 Countries with Highest Average Life Expectancy')
plt.ylabel('Life Expectancy')
plt.show()
#Insight: Countries like Japan, Switzerland, and Australia consistently rank high in life expectancy.

#7  Life Expectancy vs Schooling
sns.scatterplot(x='Schooling', y='Life_expectancy', hue='Status', data=df)
plt.title('Life Expectancy vs Average Schooling')
plt.show() #Insight: More years of schooling are generally linked to higher life expectancy.

#8. Life Expectancy vs GDP
sns.scatterplot(x='GDP', y='Life_expectancy', data=df)
plt.title('Life Expectancy vs GDP')
plt.show()#Insight: GDP has a weak but positive correlation with life expectancy, more evident in developing countries.

# 9. Life Expectancy vs BMI
sns.scatterplot(x='BMI', y='Life_expectancy', data=df)
plt.title('Life Expectancy vs BMI')
plt.show()#Insight: There is a healthy BMI range (~20-30) that associates with higher life expectancy.

#10. Immunization Impact (Polio vs Life Expectancy)
sns.scatterplot(x='Polio', y='Life_expectancy', hue='Status', data=df)
plt.title('Polio Immunization vs Life Expectancy')
plt.show()# Insight: Higher Polio immunization rates generally lead to better life expectancy outcomes.


# apply model to the dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Step 1: Prepare the data
X = df.drop(['Life_expectancy', 'Country', 'Status'], axis=1)
y = df['Life_expectancy']

# Step 2: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n📘 Linear Regression Results:")
print("R² Score:", round(r2_score(y_test, lr_pred), 4))
print("MSE:", round(mean_squared_error(y_test, lr_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, lr_pred), 4))

# Step 5: SVR
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)

print("\n⚙️ SVR Results:")
print("R² Score:", round(r2_score(y_test, svr_pred), 4))
print("MSE:", round(mean_squared_error(y_test, svr_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, svr_pred), 4))

# Step 6: Final Comparison
lr_r2 = r2_score(y_test, lr_pred)
svr_r2 = r2_score(y_test, svr_pred)

if lr_r2 > svr_r2:
    print(f"\n🏆 Best Model: Linear Regression with R² Score = {round(lr_r2, 4)}")
else:
    print(f"\n🏆 Best Model: SVR with R² Score = {round(svr_r2, 4)}")


# classification
# Binary classification: High life expectancy (>70) vs Low (≤70)
df['Life_expectancy_class'] = (df['Life_expectancy'] > 70).astype(int)

X = df.drop(columns=['Life_expectancy', 'Life_expectancy_class','Country','Status'])
y = df['Life_expectancy_class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

y_pred = knn_clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nClassification Report:\n", accuracy_score(y_test, y_pred))

