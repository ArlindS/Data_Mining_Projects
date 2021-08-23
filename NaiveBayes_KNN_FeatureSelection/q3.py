'''
    Arlind Stafaj
    Homework 3
    Data Mining 6930
    Spring 2020
    Question 3 code
'''
import sys
from scipy.io import arff
import numpy as np
import pandas as pd
import statistics

np.set_printoptions(threshold=sys.maxsize)

''' Class that takes train and test data and generates KNN algorithm on data '''


class KNN:
    # construct class with k
    def __init__(self, k=7):
        self.k = k

    # Initialize train and test data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Used to generate predictions of test data from train data using helper function _predict
    def predict(self, X):
        predicted_labels = [self._predict(i, X) for i in range(X.shape[0])]
        return np.array(predicted_labels)

    def _predict(self, i, X):
        # compute distance for each x, where x is a row in test file
        x = X[i]
        self.X_train = np.delete(self.X_train, i, 0)
        y = self.y_train[i]
        self.y_train = np.delete(self.y_train, i)
        distances = np.sqrt(np.sum((np.square((self.X_train-x))), axis=1))

        # use argsort to get the indexe values
        k_indices = np.argsort(distances)[:self.k]

        self.X_train = np.insert(self.X_train, i, x, 0)
        self.y_train = np.insert(self.y_train, i, y)
        # find the k nearest labels
        k_nearest_lbls = []
        for j in k_indices:
            if j >= i:
                a = self.y_train[j+1]
            else:
                a = self.y_train[j]
            k_nearest_lbls.append(a)

        # return the most frequent occurrence
        return statistics.mode(k_nearest_lbls)


''' Normalize Data '''


def z_score(x: np.ndarray):
    normX = []
    for column in x.T:
        n = (column - np.mean(column))/(np.std(column))
        normX.append(n)
    normX = np.array(normX)
    return normX.T
    # return (x - x.min(0)) / x.ptp(0)


# Read input data
data = arff.loadarff('veh-prime.arff')
dataf = pd.DataFrame(data[0])

# Change labels to 1,0
dataf['CLASS'] = np.where(dataf['CLASS'] == b'noncar', 0, 1)

# Get a list of feature names
features = []
for i in range(dataf.shape[1]-1):
    features.append('f' + str(i))

xtrain = dataf.iloc[:, :-1]
ytrain = dataf.iloc[:, -1]

xtrain = xtrain.to_numpy()
ytrain = ytrain.to_numpy()

'''
PART (A & B)
'''

ztrainx = z_score(xtrain)  # normalize data

# toremovetrain - used to keep track of column features left
toremovetrain = ztrainx
# toaddtrain - used to calculate which features to add
toaddtrain = np.array([])

pred = []  # store max accuracy from each test
indexes = []  # store index of column with max accuracy
greedy = True
print('\n-----------------------------------------------------')
while toremovetrain.shape[1] > 0 and greedy:
    accuracies = []
    # test performance on adding feature to select best
    for i in range(toremovetrain.shape[1]):
        clf = KNN(k=7)
        # when toaddtrain is of size zero can't use np.column_stack so initialize
        if (toaddtrain.size == 0):
            toaddtrain = np.array(ztrainx[:, i])
            toaddtrain = toaddtrain.reshape(-1, 1)
        else:
            toaddtrain = np.column_stack((toaddtrain, toremovetrain[:, i]))

        clf.fit(toaddtrain, ytrain)
        predictions = clf.predict(toaddtrain)

        # calculate accuracy of each run
        total = 0
        for n in range(len(ytrain)):
            if predictions[n] == ytrain[n]:
                total += 1
        acc = total / len(ytrain) * 100
        accuracies.append(acc)

        # remove test data
        if toaddtrain.shape[1] == 1:
            toaddtrain = np.array([])
        else:
            toaddtrain = np.delete(toaddtrain, toaddtrain.shape[1]-1, 1)

    # add best performing feature while there is a better one
    currentMax = max(accuracies) if accuracies else 0
    storedMax = max(pred) if pred else 0
    if currentMax >= storedMax:
        pred.append(max(accuracies))
        if toaddtrain.size == 0:
            toaddtrain = toremovetrain[:, np.where(
                accuracies == max(np.array(accuracies)))[0][0]]
        else:
            toaddtrain = np.column_stack((toaddtrain, toremovetrain[:, np.where(
                accuracies == max(np.array(accuracies)))[0][0]]))
        index = np.where(accuracies == max(np.array(accuracies)))[0][0]
        for i in range(ztrainx.T.shape[0]):
            if np.array_equal(toremovetrain.T[index], ztrainx.T[i]):
                indexes.append(i)
        print(indexes, '\n  percent accuracy:', max(accuracies))
        toremovetrain = np.delete(toremovetrain, index, 1)
    else:
        greedy = False

print('-----------------------------------------------------')
print('The wrapper method feature selection using Sequential Forward Selection: \n ',
      indexes[:pred.index(max(pred))+1])
print('The maximum accuracy is: ', max(pred),
      'at ', pred.index(max(pred))+1, ' iterations')
print('-----------------------------------------------------\n')
