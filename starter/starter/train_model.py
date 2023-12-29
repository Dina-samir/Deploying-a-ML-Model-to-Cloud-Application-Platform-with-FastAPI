"""
Script to train machine learning model.

Author: Dina Samir
Date: December 2023
"""

# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, compute_confusion_matrix, compute_accuracy


# load in the data.
data = pd.read_csv('../data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data['salary'])

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# train data with the process_data function
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
     training=False, encoder=encoder, lb=lb)

# Train and save a model.
modelpath = '../model/model.pkl'
encoderpath = '../model/encoder.pkl'
lbpath = '../model/lb.pkl'

# if model exits, load the model
if os.path.exists(modelpath):
    model = joblib.load(modelpath)
    encoder = joblib.load(encoderpath)
    lb = joblib.load(lbpath)

# Else Train and save a model.
else:
    model = train_model(X_train, y_train)
    joblib.dump(model, modelpath)
    joblib.dump(encoder, encoderpath)
    joblib.dump(lb, lbpath)
model = train_model(X_train, y_train)
# Compute the model's accuracy.
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print(precision, recall, fbeta)
# Compute confusion matrix
confusion_matrix = compute_confusion_matrix(y_test, y_pred, labels=list(lb.classes_))
print(confusion_matrix)

acc= compute_accuracy(y_test,y_pred)
print(acc)
