'''
    Arlind Stafaj
    Homework 3
    Data Mining 6930
    Spring 2020
    Question 2 code
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


''' Calculate z-score of data to get it normalized '''


def z_score(x: np.ndarray):
    normX = []
    for column in x.T:
        n = (column - np.mean(column))/(np.std(column))
        normX.append(n)
    normX = np.array(normX)
    return normX.T
    # return (x - x.min(0)) / x.ptp(0)


'''
PART (A)
'''
# Retrieve data information
data = arff.loadarff('veh-prime.arff')
dataf = pd.DataFrame(data[0])
dataf['CLASS'] = np.where(dataf['CLASS'] == b'noncar', 0, 1)

''' Calculate Pearson Coefficient Correlation '''


def pcc(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov_x_y = np.sum((x - mean_x) * (y - mean_y))
    stdv_x = np.sum(np.square(x - mean_x))
    stdv_y = np.sum(np.square(y - mean_y))
    correlation = cov_x_y / np.sqrt(stdv_x * stdv_y)

    return correlation


# get features of data
features = []
for i in range(dataf.shape[1]-1):
    features.append('f' + str(i))

print(len(features))
xtrain = dataf.iloc[:, :-1]
ytrain = dataf.iloc[:, -1]
xtrain = xtrain.to_numpy()
ytrain = ytrain.to_numpy()

# Compute r using the PCC method
r = []
for x in xtrain.T:
    r.append(pcc(x, ytrain))
r = np.absolute(r)
r_sorted = sorted(r, reverse=True)
indexes = []  # store indexes of r sorted
for i in range(len(features)):
    print(features[np.where(r == r_sorted[i])[0][0]], ': ',  r_sorted[i])
    indexes.append(np.where(r == r_sorted[i])[0][0])

'''
PART (B)
'''
# Get Normalized Data result for accuracy using LOOCV with KNN
ztrainx = z_score(xtrain)
predict = []


"""clf2 = KNN(k=7)
clf2.fit(ztrainx, ytrain)
predictions2 = clf2.predict(ztrainx)

# compute accuracy
total = 0
for n in range(len(ytrain)):
    if predictions2[n] == ytrain[n]:
        total += 1
print('\n', total / len(ytrain) * 100)"""

for i in range(1, len(features)):
    clf = KNN(k=7)
    # initialize with first column as ranked by r
    xtest = np.array(ztrainx[:, indexes[0]])

    # add another column in order of indexes from r
    for n in range(1, i+1):
        xtest = np.column_stack((xtest, ztrainx[:, indexes[n]]))

    clf.fit(xtest, ytrain)
    predictions = clf.predict(xtest)

    # compute accuracy
    total = 0
    for n in range(len(ytrain)):
        if predictions[n] == ytrain[n]:
            total += 1
    acc = total / len(ytrain) * 100
    predict.append(acc)

print('\nNORAMLIZED: Highest Accuracy is has ',
      predict.index(max(predict)) + 1, ' attributes. These are = ', max(predict))
print('Features for highest accuracy: ',
      indexes[:predict.index(max(predict)) + 1], '\n')
print('-----------------------------------------------------')
