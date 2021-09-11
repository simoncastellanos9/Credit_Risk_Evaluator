# Predicting Credit Risk

In this project, we built a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. 

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

We used this data to create machine learning models to classify the risk level of given loans. Specifically, to compare the Logistic Regression model and Random Forest Classifier.

## Process

### Retrieve the data

In the `Resources` folder, there are two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

We used an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

## Preprocessing: Convert categorical data to numeric

We created a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, we created a testing set from the 2020 loans, also using `pd.get_dummies()`. There are categories in the 2019 loans that do not exist in the testing set. To get around this, we used code to fill in the missing categories in the testing set. 

## Fit a LogisticRegression model and RandomForestClassifier model

We created a LogisticRegression model, fit it to the data, and printed the model's score. We then did the same for a RandomForestClassifier.

## Revisit the Preprocessing: Scale the data

The data going into these models was never scaled, an important step in preprocessing. We used `StandardScaler` to scale the training and testing sets. 

We also fitted and scored the LogisticRegression and RandomForestClassifier models on the scaled data. 

## References

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/](https://resources.lendingclub.com/)
