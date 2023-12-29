"""
Test cases for the fastapi in main.py

Author: Dina Samir
Date: December, 2023
"""
import requests
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
print(response.status_code)
print(response.text )

