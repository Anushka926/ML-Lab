import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("car prices.csv")

# -----------------------------
# Basic Information
# -----------------------------
print(df.head())

print("Shape:", df.shape)
print(df.columns)
print(df.info())
print(df.describe())

print(df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())
print(df.dtypes)

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

# Distribution of Selling Price
plt.figure(figsize=(8,5))
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Distribution of Car Prices")
plt.show()

# Brand-wise Selling Price
plt.figure(figsize=(12,5))
sns.boxplot(x="Brand", y="Selling_Price", data=df)
plt.xticks(rotation=90)
plt.title("Brand-wise Selling Price")
plt.show()

# Fuel Type Count
plt.figure(figsize=(6,4))
sns.countplot(x="Fuel", data=df)
plt.title("Fuel Type Count")
plt.show()

# Transmission Count
plt.figure(figsize=(6,4))
sns.countplot(x="Transmission", data=df)
plt.title("Transmission Count")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x="KM_Driven", y="Selling_Price", data=df)
plt.title("KM Driven vs Selling Price")
plt.show()

# -----------------------------
# Correlation Matrix
# -----------------------------
temp = df.copy()

le = LabelEncoder()

for col in temp.select_dtypes(include="object"):
    temp[col] = le.fit_transform(temp[col])

plt.figure(figsize=(8,6))
sns.heatmap(temp.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# Data Preprocessing
# -----------------------------
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Simple Linear Regression
# -----------------------------
X_simple = df[["Year"]]
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X_simple,
    y,
    test_size=0.2,
    random_state=42
)

slr = LinearRegression()
slr.fit(X_train, y_train)

y_pred = slr.predict(X_test)

print("Simple Linear Regression")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score :", r2_score(y_test, y_pred))

# Regression Line
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color="red")
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.title("Simple Linear Regression")
plt.show()

# -----------------------------
# Multiple Linear Regression
# -----------------------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMultiple Linear Regression")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score :", r2_score(y_test, y_pred))

# -----------------------------
# Feature Selection
# -----------------------------
selector = SelectKBest(score_func=f_regression, k=5)

X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]

print("\nSelected Features:")
print(selected_features)

X = df[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

feature_model = LinearRegression()
feature_model.fit(X_train, y_train)

y_pred = feature_model.predict(X_test)

print("\nAfter Feature Selection")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score :", r2_score(y_test, y_pred))

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
parameters = {
    "fit_intercept": [True, False],
    "positive": [True, False]
}

grid = GridSearchCV(
    LinearRegression(),
    parameters,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("\nBest Model Performance")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score :", r2_score(y_test, y_pred))
