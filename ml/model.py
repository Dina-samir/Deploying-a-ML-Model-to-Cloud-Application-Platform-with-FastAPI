"""
Script contains main functions to train machine learning and compute model metrics.

Author: Dina Samir
Date: December 2023
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix , accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1,verbose=2)
    rf.fit(X_train, y_train)

    return rf

    


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
    
def compute_confusion_matrix(y, y_pred, labels=None):
    """
    Compute confusion matrix 

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    cm : np.array
    """
    cm = confusion_matrix(y, y_pred)
    return cm


def compute_accuracy(y_true,y_pred):
    """
    Compute model accuracy

    Inputs
    ------
    y_true : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    acc : float
    """
    acc= accuracy_score(y_true, y_pred) *100
    return acc