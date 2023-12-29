"""
Test cases for the model in model.py

Author: Dina Samir
Date: December, 2023
"""
import pandas as pd
import numpy as np
from starter.ml.model import train_model, compute_model_metrics
import os 

def test_model_existance():
    """
    Test if model is exist in path.
    """
    modelpath = './model/model.pkl'
    assert os.path.exists(modelpath), f"Model path '{modelpath}' does not exist!"


def test_load_data():
    """
    Check the existance od the data after cleaning
    """
    data = pd.read_csv("data/census_clean.csv")
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0]>0 , f"the data have no rows"
    assert data.shape[1]>0 , f"the data have no columns"


def test_output_type():
    """
    Test the type of the output.
    """
    y1 = np.random.randint(0, 2, 10)
    preds1 = np.random.randint(0, 2, 10)
    precision1, recall1, fbeta1 = compute_model_metrics(y1, preds1)
    assert isinstance(precision1, float), f"precision1 is not float"
    assert isinstance(recall1, float), f"recall1 is not float"
    assert isinstance(fbeta1, float), f"fbeta1 is not float"       

test_model_existance()
test_load_data()
test_output_type()