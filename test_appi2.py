import requests


'''
after run app by the following command in terminal (python -m uvicorn main:app)
run this script
'''

# test welcome:
def test_get():
      response = requests.get("http://127.0.0.1:8000/welcome/")
      assert response.status_code == 200
      assert response.json() == {"Welcome to the API!"}

def test_post1():
      response = requests.post("http://127.0.0.1:8000/inference/", 
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
      assert response.json() == {"salary": "<=50K"}
      

def test_post2():
      response = requests.post("http://127.0.0.1:8000/inference/", 
                  json={
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

      print(response.text)
      assert response.status_code == 200
      assert response.json() == {"salary": "<=50K"}
      