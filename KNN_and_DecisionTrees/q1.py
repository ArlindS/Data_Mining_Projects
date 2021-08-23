'''
    Arlind Stafaj 
    Homework 2 
    Data Mining 6930 
    Spring 2020 
    Question 1 code 
'''
import sys
from collections import Counter
import numpy as np
import pandas as pd
from math import sqrt

np.set_printoptions(threshold=sys.maxsize)

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
# Drop Columns from testData DataFrame
ids = testx.iloc[:50, :1]
testx.drop(testx.columns[0], axis=1, inplace=True)
# New Data Frame for training and test labels
trainy = trainx[['class']].copy()
testy = testx[['Label']].copy()
trainy = trainy.to_numpy()
testy = testy.to_numpy()

# Drop label columns from trainData and testData
trainx.drop('class', axis=1, inplace=True)
testx.drop('Label', axis=1, inplace=True)
trainx = trainx.to_numpy()
testx = testx.to_numpy()

a = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
'''
PART(a)
'''

print('\nQUESTION 1: \n PART (a): \n')
for i in a:
    clf = KNN(k=i)
    clf.fit(trainx, trainy)
    predictions = clf.predict(testx)
    total = 0
    for n in range(len(predictions)):
        if predictions[n] == testy[n]:
            total += 1
    acc = total / len(testy) * 100
    print(' K = ', i, '\n Percent accuracy = ',
          acc, '\n-----------------------------------------')

'''
    PART (b)
'''

# calculate z_score to normalize data for train and test


def z_score(x):
    normX = []
    for column in x.T:
        n = (column - np.mean(column))/np.std(column)
        normX.append(n)
    normX = np.array(normX)
    return normX.T


ztrainx = z_score(trainx)
ztestx = z_score(testx)

print('\nQUESTION 1: \n PART (b): \n')

predict_50 = []

for i in a:
    clf = KNN(k=i)
    clf.fit(ztrainx, trainy)
    predictions = clf.predict(ztestx)
    predict_50.append(predictions)
    total = 0
    for n in range(len(predictions)):
        if predictions[n] == testy[n]:
            total += 1
    acc = total / len(testy) * 100
    print(' K = ', i, '\n Percent accuracy = ',
          acc, '\n-----------------------------------------')

'''
    PART (c)
'''
predict_50 = np.array(predict_50).T
predict_50 = predict_50[:50, :]
predict_df = pd.DataFrame(predict_50, index=ids, columns=a)
predict_df.replace([1, 0], ['spam', 'NO'], inplace=True)
print('\n', predict_df)
