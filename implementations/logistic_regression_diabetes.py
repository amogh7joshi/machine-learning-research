#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from regression.logistic.logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as LogisticRegressionSKLearn

# An implementation of the Logistic Regression algorithm on the Pima Indians Diabetes Dataset.
# Compared to the Scikit-Learn Logistic Regression implementation,
# it attains accuracy within a margin of 1% and a confusion matrix of only 1 difference.

# Create the dataset from the CSV file.
dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'diabetes.csv'))

# Preprocess the dataset (convert to X/y and apply scaling).
X = dataset.drop(['Outcome'], axis = 1)
X = StandardScaler().fit_transform(X)
y = dataset['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit the logistic regression to the dataset.
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train, l2_coef = 1)

# Get the predictions of the model on the test set.
y_pred = logistic_regression.predict(X_test)

# Create the sklearn version of Logistic Regression.
logistic_regression_sklearn = LogisticRegressionSKLearn()
logistic_regression_sklearn.fit(X_train, y_train)

# Evaluate the model (on accuracy).
print(accuracy_score(logistic_regression.predict(X_test), y_test))
print(accuracy_score(logistic_regression_sklearn.predict(X_test), y_test))

# Create Confusion Matrices for evaluation.
print(confusion_matrix(logistic_regression.predict(X_test), y_test))
print(confusion_matrix(logistic_regression_sklearn.predict(X_test), y_test))






