# Model Card
Model Developed by : Dina Samir
Date               : December, 2023 Version: 1.0.0
Type               : Binary Clasifier
Dataset            : https://archive.ics.uci.edu/ml/datasets/census+income

## Model Details
The model is a Random Forest classifier with n_estimators=100, max_depth=15, random_state=42, n_jobs=-1 and verbose=2.


## Intended Use
Predict the income of a person in two clasess > greater than $50K or less than $50K.

## Training Data
split the following dataset:[https://archive.ics.uci.edu/ml/datasets/census+income] to train 80% od data and test data 20%

## Evaluation Data
- The shape of the data is (32561, 15)
- Apply data cleaning 
- - remove ? from rows
- - remove null rows
- - remove spaces from column names
- After data cleaning the shape of data became (30162, 15)
- The data was splitted into 80% training and 20% testing data.
- The model was trained on the training data and evaluated on the testing data.

## Metrics
The model was evaluated using precision, recall, fbeta and accuracy.
- precision : 0.79
- recall    : 0.57 
- fbeta     : 0.67
- accuracy  :86%

Confusion matrix values
[[4294  237]
 [ 633  869]]


## Ethical Considerations
This dataset is not a reliable reflection of the distribution of salaries and should not be used to make assumptions about the salary levels of particular groups of people.

## Caveats and Recommendations
implement hyperparameter tuning in model traing.
use K-fold cross validation instead of a train-test split.
