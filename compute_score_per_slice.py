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

def compute_score_per_slice(test_df, categorical_feature, y, y_pred):
    """
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed
    ------
    test_df: test dataframe pre-processed with features as column used for slices
    categorical_feature:feature on which to perform the slices
    y : np.arraycorresponding known labels, binarized.
    y_pred : np.arrayPredicted labels, binarized
    Returns
    ------
    Dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """    
    slice_options = test_df[categorical_feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options, 
                            columns=['categorical_feature','n_samples','precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = test_df[categorical_feature]==option

        slice_y = y[slice_mask]
        slice_preds = y_pred[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)


    return len(slice_y), precision, recall,fbeta

data = pd.read_csv('./data/census_clean.csv')
# Train and save a model.
modelpath = './model/model.pkl'
encoderpath = './model/encoder.pkl'
lbpath = './model/lb.pkl'

model = joblib.load(modelpath)
encoder = joblib.load(encoderpath)
lb = joblib.load(lbpath)

train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=42, 
                                stratify=data['salary']
                                )
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
# evaluate trained model on test set
preds = inference(model, X_test)


with open("slice_output.txt", "w") as output_file:
    for feature in cat_features:
        slice_len, precision, recall,fbeta = compute_score_per_slice(test, feature, y_test, preds)
        output_file.write(f"Performance Metrics for '{feature}':\n")
        output_file.write(f"Precision: {precision:.4f}\n")
        output_file.write(f"Recall: {recall:.4f}\n")
        output_file.write(f"fbeta: {fbeta:.4f}\n")
        output_file.write("=" * 40 + "\n")
