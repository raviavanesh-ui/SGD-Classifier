# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: AVANESH R
RegisterNumber: 25018356


# Step 1: Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data        # Features
y = iris.target      # Labels (species)


# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Step 4: Feature Scaling (very important for SGD)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 5: Create the SGD Classifier
sgd_clf = SGDClassifier(
    loss="log_loss",     # Logistic Regression
    max_iter=1000,
    learning_rate="optimal",
    random_state=42
)


# Step 6: Train the model
sgd_clf.fit(X_train, y_train)


# Step 7: Make predictions
y_pred = sgd_clf.predict(X_test)


# Step 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Step 9: Predict a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example flower
sample_scaled = scaler.transform(sample)
prediction = sgd_clf.predict(sample_scaled)

print("\nPredicted Species:", iris.target_names[prediction][0])




*/
```

## Output:![WhatsApp Image 2025-12-26 at 8 20 14 AM](https://github.com/user-attachments/assets/b8f79c4c-1851-4be6-bf4d-f6cdc09f53d2)




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
