# Lab 7: Naive Bayes Classification on the Raisin Dataset
# Name: Anushka Boreklar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

pd.set_option('display.max_columns', None)

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
df = pd.read_excel('/content/Raisin_Dataset.xlsx')   # change path if running locally
print(df.head())

# ------------------------------------------------------------------
# 2. Encode target label (Besni = 0, Kecimen = 1)
# ------------------------------------------------------------------
label_encoder = LabelEncoder()
df['Class_encoded'] = label_encoder.fit_transform(df['Class'])
print(df[['Class', 'Class_encoded']].head(5))
print("Unique encoded classes:",
      dict(zip(label_encoder.classes_,
               label_encoder.transform(label_encoder.classes_))))

# ------------------------------------------------------------------
# 3. Features and target
# ------------------------------------------------------------------
numerical_features = [c for c in df.select_dtypes(include=np.number).columns
                      if c != 'Class_encoded']
print("Numerical features:", numerical_features)

X = df[numerical_features]
y = df['Class_encoded']
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# ------------------------------------------------------------------
# 4. Train/test split (80:20)
# ------------------------------------------------------------------
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ------------------------------------------------------------------
# 5. Train Gaussian Naive Bayes
# ------------------------------------------------------------------
gnb_model = GaussianNB()
gnb_model.fit(X_train_r, y_train_r)

# ------------------------------------------------------------------
# 6. Predict on the test set
# ------------------------------------------------------------------
y_pred = gnb_model.predict(X_test_r)
print("Predicted labels:")
print(y_pred)

# ------------------------------------------------------------------
# 7. Actual vs predicted comparison
# ------------------------------------------------------------------
comparison = pd.DataFrame({'Actual': y_test_r, 'Predicted': y_pred})
comparison["Actual class"] = label_encoder.inverse_transform(comparison["Actual"])
comparison["Predicted class"] = label_encoder.inverse_transform(comparison["Predicted"])
print(comparison.head(5))

# ------------------------------------------------------------------
# 8. Evaluation metrics
# ------------------------------------------------------------------
accuracy  = accuracy_score(y_test_r, y_pred)
precision = precision_score(y_test_r, y_pred, average='weighted')
recall    = recall_score(y_test_r, y_pred, average='weighted')
f1        = f1_score(y_test_r, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\n======MODEL PERFORMANCE===========")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification report:\n")
print(classification_report(y_test_r, y_pred))

# ------------------------------------------------------------------
# 9. Training vs testing accuracy
# ------------------------------------------------------------------
y_train_pred = gnb_model.predict(X_train_r)
y_test_pred  = gnb_model.predict(X_test_r)
train_accuracy = accuracy_score(y_train_r, y_train_pred)
test_accuracy  = accuracy_score(y_test_r, y_test_pred)   # recalculated here
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# ------------------------------------------------------------------
# 10. Exploratory Data Analysis
# ------------------------------------------------------------------
# 10.1 Box plot of features grouped by class
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.melt(id_vars='Class',
                         value_vars=['Area', 'MajorAxisLength', 'MinorAxisLength',
                                     'ConvexArea', 'Perimeter']),
            x='variable', y='value', hue='Class')
plt.title('Box Plot of Selected Numerical Features by Raisin Class')
plt.xlabel('Features')
plt.ylabel('Values')
plt.yscale('log')   # Area and ConvexArea are much larger than the other features
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 10.2 Bar chart of class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Class', palette='viridis', hue='Class', legend=False)
plt.title('Bar Chart - Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 10.3 Histogram of MajorAxisLength by class
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='MajorAxisLength', hue='Class', kde=True, bins=30)
plt.title('Histogram - Distribution of MajorAxisLength')
plt.xlabel('Major Axis Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 10.4 Correlation heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap - Feature Correlation Matrix')
plt.tight_layout()
plt.show()
