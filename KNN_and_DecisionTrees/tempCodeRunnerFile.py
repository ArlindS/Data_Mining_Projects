'''
    Arlind Stafaj 
    Homework 2 
    Data Mining 6930 
    Spring 2020 
    Question 1 code 
'''
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt


''' Function to compute eculidean distance '''


def euclidean_dist(row1, row2):
    return np.sqrt(np.sum((row1-row2)**2))


''' Class that takes train and test data and generates KNN algorithm on data '''


class KNN:
    # construct class with k
    def __init__(self, k=3):
        self.k = k

    # Initialize train and test data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Used to generate predictions of test data from train data using helper function _predict
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distance for each x, where x is a row in test file
        distances = [euclidean_dist(x, x_train) for x_train in self.X_train]
        # use argsort to get the indexe values
        k_indices = np.argsort(distances)[:self.k]

        # find the k nearest labels
        k_nearest_lbls = []
        for i in k_indices:
            a = self.y_train[i]
            k_nearest_lbls.append(a[0])

        # return the most frequent occurrence
        most_common = Counter(k_nearest_lbls).most_common(1)
        return most_common[0][0]


# Read train and test data
trainx = pd.read_csv("spam_train.csv")
testx = pd.read_csv("spam_test.csv")
print(testx)
# Drop Columns from testData DataFrame
ids = testx.iloc[:50, :1]
print(ids)