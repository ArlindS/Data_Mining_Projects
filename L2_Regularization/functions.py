import pandas as pd
import numpy as np
from copy import deepcopy

'''
    Weighted Matrix Function: w = ((X^T * X) + (lambda * I))^-1 * (X^T * y)
'''


def wMatrix(x: np.ndarray, y: np.ndarray, l):
    i = np.identity(x.shape[1])
    w = (np.linalg.inv((x.T.dot(x) + (l * i)))).dot((x.T.dot(y)))
    return w


'''
    MSE Function: E(w) = ((y - (X * w))^2) / N
'''


def MSE_result(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    MSE = np.sum((y-(x.dot(w)))**2) / np.shape(x)[0]
    return MSE
# Question 1(a)


'''
    Generates results for mse values of test and training data.
    train - is variable for training file name 
    test - is variable for test file name 
'''
'''
Question 1(a)
'''


def MSE_generator(train, test):
    # Read Excel files as data frames using pandas and convert them to arrays with numpy
    train_data: np.ndarray = pd.read_excel(train)
    test_data: np.ndarray = pd.read_excel(test)

    # Split train data into matrix X and Y and add column of 1's in X
    train_x = train_data.iloc[:, :-1]
    train_x.insert(0, 'Default', 1)
    train_y = train_data.iloc[:, -1]

    # Split test data into matrix X and Y and add column of 1's in X
    test_x = test_data.iloc[:, :-1]
    test_x.insert(0, 'Default', 1)
    test_y = test_data.iloc[:, -1]

    # Lists for holding MSE values for train and test data
    train_mse = []
    test_mse = []

    # A for loop where wighted matrix is calculated and used to get MSE results
    for i in range(151):
        w = wMatrix(train_x, train_y, i)
        tr_mse = MSE_result(train_x, train_y, w)
        te_mse = MSE_result(test_x, test_y, w)
        train_mse.append(tr_mse)
        test_mse.append(te_mse)

    return train_mse, test_mse


'''
Question 1(b)
'''


def MSE_generator_2(train, test):
    # Read Excel files as data frames using pandas and convert them to arrays with numpy
    train_data: np.ndarray = pd.read_excel(train)
    test_data: np.ndarray = pd.read_excel(test)

    # Split train data into matrix X and Y and add column of 1's in X
    train_x = train_data.iloc[:, :-1]
    train_x.insert(0, 'Default', 1)
    train_y = train_data.iloc[:, -1]

    # Split test data into matrix X and Y and add column of 1's in X
    test_x = test_data.iloc[:, :-1]
    test_x.insert(0, 'Default', 1)
    test_y = test_data.iloc[:, -1]

    # Lists for holding MSE values for train and test data
    train_mse = [0]
    test_mse = [0]

    # A for loop where wighted matrix is calculated and used to get MSE results
    for i in range(1, 151):
        w = wMatrix(train_x, train_y, i)
        tr_mse = MSE_result(train_x, train_y, w)
        te_mse = MSE_result(test_x, test_y, w)
        train_mse.append(tr_mse)
        test_mse.append(te_mse)

    return train_mse, test_mse


'''
Question 2
'''


def generate_train_data(train):
    # Read Excel files as data frames using pandas and convert them to arrays with numpy
    train_data: np.ndarray = pd.read_excel(train)
    train_data.insert(0, 'Default', 1)
    train_folds = np.split(train_data, 10)
    train_x = train_data.iloc[:, :-1]
    train_y = train_data.iloc[:, -1]

    return train_folds, train_x, train_y


def generate_test_data(test):
    # Read Excel files as data frames using pandas and convert them to arrays with numpy
    test_data: np.ndarray = pd.read_excel(test)
    test_x = test_data.iloc[:, :-1]
    test_x.insert(0, 'Default', 1)
    test_y = test_data.iloc[:, -1]

    return test_x, test_y


def cv_results(train, test):
    test_mse = list(range(151))
    train_folds, train_x, train_y = generate_train_data(train)
    test_x, test_y = generate_test_data(test)

    for l in range(151):
        test_mse[l] = 0
        for i in range(10):
            temp_folds = deepcopy(train_folds)

            # Use the i-th fold as the testing data
            x_test = np.array(temp_folds[i])
            x_test = x_test[:, :-1]

            y_test = np.array(temp_folds.pop(i))
            y_test = y_test[:, -1].T

            # Use the rest of the folds for training
            x_train = np.vstack(temp_folds)[:, :-1]
            y_train = np.vstack(temp_folds)[:, -1].T

            # Calculate the w matrix with the training data.
            w = wMatrix(x_train, y_train, l)

            # Sum the MSE for the given lambda value
            test_mse[l] += MSE_result(x_test, y_test, w)

        # Average the MSE
        test_mse[l] /= 10

    print("\n****** 10-Fold CV with: train data: ",
          train, "|| test data: ", test, "\n")
    print("    Minimum MSE values from 10-Fold Cross Validation: ",
          min(test_mse), "\n    Lamda: ", test_mse.index(min(test_mse)), '\n\n')
    w1 = wMatrix(train_x, train_y, test_mse.index(min(test_mse)))
    print("    Retraining with previous selected lambda value: ", test_mse.index(
        min(test_mse)), "\n    Test MSE from retraining: ", MSE_result(test_x, test_y, w1), '\n')
