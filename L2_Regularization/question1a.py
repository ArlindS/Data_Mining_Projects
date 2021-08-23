'''
    Arlind Stafaj HW1 
    Question 1(a)
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functions import MSE_generator, MSE_result, wMatrix

# List of Lamda values
l_values = list(range(151))


print("\nQuestion 1(a)")
print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")
print("L2 Regularization with train-100-10.xlsx vs. test-100-10.xlsx\n")
train_1, test_1 = MSE_generator('train-100-10.xlsx', 'test-100-10.xlsx')
print("Train: ")
print("MSE: ", min(train_1), "\nlamda: ", train_1.index(min(train_1)))
print("\nTest: ")
print("MSE: ", min(test_1), "\nlamda: ", test_1.index(min(test_1)))
print('\n\n')

print("-------------------------------------------------------------------")
print("L2 Regularization with train-100-100.xlsx vs. test-100-100.xlsx\n")
train_2, test_2 = MSE_generator('train-100-100.xlsx', 'test-100-100.xlsx')
print("Train: ")
print("MSE: ", min(train_2), "\nlamda: ", train_2.index(min(train_2)))
print("\nTest: ")
print("MSE: ", min(test_2), "\nlamda: ", test_2.index(min(test_2)))
print('\n\n')

print("-------------------------------------------------------------------")
print("L2 Regularization with train-1000-100.xlsx vs. test-1000-100.xlsx\n")
train_3, test_3 = MSE_generator('train-1000-100.xlsx', 'test-1000-100.xlsx')
print("Train: ")
print("MSE: ", min(train_3), "\nlamda: ", train_3.index(min(train_3)))
print("\nTest: ")
print("MSE: ", min(test_3), "\nlamda: ", test_3.index(min(test_3)))
print('\n\n')

print("-------------------------------------------------------------------")
print("L2 Regularization with train-50(1000)-100.xlsx vs. test-1000-100.xlsx\n")
train_4, test_4 = MSE_generator(
    'train-50(1000)-100.xlsx', 'test-1000-100.xlsx')
print("Train: ")
print("MSE: ", min(train_4), "\nlamda: ", train_4.index(min(train_4)))
print("\nTest: ")
print("MSE: ", min(test_4), "\nlamda: ", test_4.index(min(test_4)))
print('\n\n')

print("-------------------------------------------------------------------")
print("L2 Regularization with train-100(1000)-100.xlsx vs. test-1000-100.xlsx\n")
train_5, test_5 = MSE_generator(
    'train-100(1000)-100.xlsx', 'test-1000-100.xlsx')
print("Train: ")
print("MSE: ", min(train_5), "\nlamda: ", train_5.index(min(train_5)))
print("\nTest: ")
print("MSE: ", min(test_5), "\nlamda: ", test_5.index(min(test_5)))
print('\n\n')

print("-------------------------------------------------------------------")
print("L2 Regularization with train-150(1000)-100.xlsx vs. test-1000-100.xlsx\n")
train_6, test_6 = MSE_generator(
    'train-150(1000)-100.xlsx', 'test-1000-100.xlsx')
print("Train: ")
print("MSE: ", min(train_6), "\nlamda: ", train_6.index(min(train_6)))
print("\nTest: ")
print("MSE: ", min(test_6), "\nlamda: ", test_6.index(min(test_6)))
print('\n\n')

print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")

'''
    Graphs
'''
fig = plt.figure(figsize=(18, 8))

plt.subplot(2, 3, 1)
plt.plot(l_values, train_1)
plt.plot(l_values, test_1)
plt.title('train-100-10 vs. test-100-10')
plt.legend(['train-100-10', 'test-100-10'])
plt.xlabel('lamda')
plt.ylabel('MSE')

plt.subplot(2, 3, 2)
plt.plot(l_values, train_2)
plt.plot(l_values, test_2)
plt.title('train-100-100 vs. test-100-100')
plt.legend(['train-100-100', 'test-100-100'])
plt.xlabel('lamda')
plt.ylabel('MSE')

plt.subplot(2, 3, 3)
plt.plot(l_values, train_3)
plt.plot(l_values, test_3)
plt.title('train-1000-100 vs. test-1000-100')
plt.legend(['train-1000-100', 'test-1000-100'])
plt.xlabel('lamda')
plt.ylabel('MSE')

plt.subplot(2, 3, 4)
plt.plot(l_values, train_4)
plt.plot(l_values, test_4)
plt.title('train-50(1000)-100 vs. test-1000-100')
plt.legend(['train-(50)1000-100', 'test-1000-100'])
plt.xlabel('lamda')
plt.ylabel('MSE')

plt.subplot(2, 3, 5)
plt.plot(l_values, train_5)
plt.plot(l_values, test_5)
plt.title('train-(100)1000-100 vs. test-1000-100')
plt.legend(['train-(100)1000-100', 'test-1000-100'])
plt.xlabel('lamda')
plt.ylabel('MSE')

plt.subplot(2, 3, 6)
plt.plot(l_values, train_6)
plt.plot(l_values, test_6)
plt.title('train-(150)1000-100 vs. test-1000-100')
plt.legend(['train-(150)1000-100', 'test-1000-100'])
plt.xlabel('lamda')
plt.ylabel('MSE')

plt.subplots_adjust(left=.1, bottom=.1, right=.95,
                    top=0.95, wspace=.2, hspace=.4)
plt.show()
fig.savefig("Question1(a).png")
