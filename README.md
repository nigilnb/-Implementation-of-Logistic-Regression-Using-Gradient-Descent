# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset and encode all categorical features using Label Encoding.
2.Split the dataset into training and testing sets and apply feature scaling.
3.Initialize weights and bias to zero and define the sigmoid activation function.
4.For a fixed number of epochs, compute predictions and calculate gradients of weights and bias.
5.Update the weights and bias using gradient descent to minimize the loss.
6.Predict class labels for test data and evaluate the model using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: NIGIL.S
RegisterNumber:212225240100
*/
```
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("C:/Users/acer/Downloads/Placement_Data (1).csv")

encode_cols = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

for col in encode_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

df.drop(['sl_no', 'salary'], axis=1, inplace=True)

X = df.drop('status', axis=1).values
y = df['status'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weights = np.zeros((X_train.shape[1], 1))
bias = 0

lr = 0.01
epochs = 1000
m = X_train.shape[0]

for _ in range(epochs):
    z = np.dot(X_train, weights) + bias
    y_hat = sigmoid(z)

    dw = (1/m) * np.dot(X_train.T, (y_hat - y_train))
    db = (1/m) * np.sum(y_hat - y_train)

    weights -= lr * dw
    bias -= lr * db

    z_test = np.dot(X_test, weights) + bias


y_pred = sigmoid(z_test)
y_pred = (y_pred >= 0.5).astype(int).ravel()

print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test, y_pred))

print("\nACCURACY:")
print(accuracy_score(y_test, y_pred))

print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))


## Output:
<img width="468" height="343" alt="image" src="https://github.com/user-attachments/assets/bf1144ec-a41a-435e-97b1-af7df107869c" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

