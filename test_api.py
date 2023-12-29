"""
Test cases for the fastapi in main.py

Author: Dina Samir
Date: December, 2023
"""
import requests

# test welcome:
def test_get():
      response = requests.get("https://salary-prediction-e6q7.onrender.com/")
      assert response.status_code == 200
      assert response.json() == ["Welcome to the API!"]

def test_post1():
      response = requests.post("https://salary-prediction-e6q7.onrender.com/inference/", 
                  json={"age": 40,
                        "workclass": "Private",
                        "fnlgt": 176609,
                        "education": "Some-college",
                        "education-num": 8,
                        "marital-status": "Married-civ-spouse",
                        "occupation": "Exec-managerial",
                        "relationship": "Not-in-family",
                        "race": "Black",
                        "sex": "Male",
                        "capital-gain": 0,
                        "capital-loss": 0,
                        "hours-per-week": 80,
                        "native-country": "United-States"})

      print(response.text)
      assert response.status_code == 200
      assert response.json() == ["<=50K","salary"]

def test_post2():    
    response = requests.post("https://salary-prediction-e6q7.onrender.com/inference/",
                            json={"age": 45,
                            "workclass": "Private",
                            "fnlgt": 45781,
                            "education": "Masters",
                            "education-num": 14,
                            "marital-status": "Married-civ-spouse",
                            "occupation": "Prof-specialty",
                            "relationship": "Not-in-family",
                            "race": "White",
                            "sex": "Male",
                            "capital-gain": 14084,
                            "capital-loss": 0,
                            "hours-per-week": 40,
                            "native-country": "United-States"})
    assert response.status_code == 200
    assert response.json() == [">50K","salary"]
