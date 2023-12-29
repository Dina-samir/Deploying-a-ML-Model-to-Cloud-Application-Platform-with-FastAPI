"""
Test cases for the fastapi in main.py

Author: Dina Samir
Date: December, 2023
"""
from fastapi.testclient import TestClient
from main import app

# Instantiate the client for testing.
client = TestClient(app)


def test_get():
    """ Test root function will get a succesful response"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Welcome to the API!"}



def test_post1():
    """Test predict funtion when income is less than 50K"""

    response = client.post("/predict-income", json={
        "age": 30,
        "workclass": "Private",
        "fnlgt": 183175,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })
    assert response.status_code == 200
    assert response.json() == {"salary": "<=50K"}
    

def test_post2():
    """Test predict funtion when income is more than 50K"""

    response = client.post("/predict-income", json={
        "age": 40,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 100,
        "native_country": "United-States"
    })
    assert response.status_code == 200
    assert response.json() == {"salary": ">50K"}