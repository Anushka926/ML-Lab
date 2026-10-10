# Lab 6: To Implement the Perceptron Learning Algorithm
# Student: Anushka Borekar | Section B | Roll No. 37 | Batch B3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
print("Dataset shape:", X.shape)
print("Feature names:", iris.feature_names)
print("Class names:", iris.target_names)
print("Class distribution:", np.bincount(y))

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 3. Standardize features using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and train the Perceptron model
model = Perceptron(max_iter=1000, eta0=0.1, random_state=42, tol=1e-3)
model.fit(X_train_scaled, y_train)

# 5. Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\\nPerceptron test accuracy: {:.2%}".format(accuracy))
print("\\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("Perceptron - Iris Test Confusion Matrix")
plt.tight_layout()
plt.show()

# 7. Predict the class of a new flower sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_scaled = scaler.transform(new_sample)
predicted_class = model.predict(new_sample_scaled)[0]
print("New sample:", new_sample.tolist()[0])
print("Predicted class:", iris.target_names[predicted_class])
