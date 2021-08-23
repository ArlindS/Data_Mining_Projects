'''
    Arlind Stafaj HW1
    Question 3
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functions import MSE_generator, MSE_result, wMatrix
from random import sample


fig = plt.figure(figsize=(18, 10))

train_data = pd.read_excel('train-1000-100.xlsx')
train_data.insert(0, 'Default', 1)
train_x = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, -1]
train_x = train_x.to_numpy()
train_y = train_y.to_numpy()

test_data = pd.read_excel('test-1000-100.xlsx')
test_x = test_data.iloc[:, :-1]
test_x.insert(0, 'Default', 1)
test_y = test_data.iloc[:, -1]
test_x = test_x.to_numpy()
test_y = test_y.to_numpy()

lambdas = [1, 25, 150]
for l in lambdas:
    mse_test = []
    for a in range(10, 1000, 10):
        listTest = list(range(20))

        # 20 trials of each
        for i in range(20):
            # data structures to hold subset data
            subset_x: np.ndarray = np.empty([a, 101])
            subset_y: np.ndarray = np.empty([a])

            # random numbers generated as indexes from training data to be chosen
            samples = sample(range(1, train_data.shape[0]), a)

            # Create Subset data of increasing size
            for j in range(a):
                subset_x[j] = train_x[samples[j]-1:samples[j]]
                subset_y[j] = train_y[samples[j]]
            w = wMatrix(subset_x, subset_y, l)
            listTest[i] = (MSE_result(test_x, test_y, w))

        # Get average of each trial
        mse_test.append(sum(listTest) / len(listTest))
    print("Average MSE Values: \n")
    print(mse_test, '\n')

    # Plot the average MSEs for each sample
    plt.subplot(1, 3, lambdas.index(l)+1)
    plt.plot(mse_test)
    plt.title("Learning Curve for Lambda = {}".format(l))
    plt.xlabel("Training Set Sample Size")
    plt.ylabel("MSE")


plt.subplots_adjust(left=.1, bottom=.3, right=.95,
                    wspace=.2, hspace=0)
plt.show()
fig.savefig("Plot_Question3.png")
