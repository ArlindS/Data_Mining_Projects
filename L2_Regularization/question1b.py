'''
    Arlind Stafaj HW1 
    Question 1(b)
'''
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functions import MSE_generator_2, MSE_result, wMatrix

l_values = list(range(1, 151))

print("-------------------------------------------------------------------")
print("L2 Regularization with train-100-100.xlsx vs. test-100-100.xlsx\n")
train_2b, test_2b = MSE_generator_2('train-100-100.xlsx', 'test-100-100.xlsx')
print("Train: ")
print("MSE: ", min(train_2b[1:]), "\nlamda: ",
      train_2b.index(min(train_2b[1:])))
print("\nTest: ")
print("MSE: ", min(test_2b[1:]), "\nlamda: ", test_2b.index(min(test_2b[1:])))
print('\n\n')

print("-------------------------------------------------------------------")
print("L2 Regularization with train-50(1000)-100.xlsx vs. test-1000-100.xlsx\n")
train_4b, test_4b = MSE_generator_2(
    'train-50(1000)-100.xlsx', 'test-1000-100.xlsx')
print("Train: ")
print("MSE: ", min(train_4b[1:]), "\nlamda: ",
      train_4b.index(min(train_4b[1:])))
print("\nTest: ")
print("MSE: ", min(test_4b[1:]), "\nlamda: ", test_4b.index(min(test_4b[1:])))
print('\n\n')

print("-------------------------------------------------------------------")
print("L2 Regularization with train-100(1000)-100.xlsx vs. test-1000-100.xlsx\n")
train_5b, test_5b = MSE_generator_2(
    'train-100(1000)-100.xlsx', 'test-1000-100.xlsx')
print("Train: ")
print("MSE: ", min(train_5b[1:]), "\nlamda: ",
      train_5b.index(min(train_5b[1:])))
print("\nTest: ")
print("MSE: ", min(test_5b[1:]), "\nlamda: ", test_5b.index(min(test_5b[1:])))
print('\n\n')

''' 
    Graphs 
'''
fig = plt.figure(figsize=(18, 8))

plt.subplot(1, 3, 1)
plt.plot(l_values, train_2b[1:])
plt.plot(l_values, test_2b[1:])
plt.title('train-100-100 vs. test-100-100')
plt.legend(['train-100-100', 'test-100-100'])
plt.xlabel('lamda')
plt.ylabel('MSE')

plt.subplot(1, 3, 2)
plt.plot(l_values, train_4b[1:])
plt.plot(l_values, test_4b[1:])
plt.title('train-50(1000)-100 vs. test-1000-100')
plt.legend(['train-50(1000)-100', 'test-1000-100'])
plt.xlabel('lamda')
plt.ylabel('MSE')

plt.subplot(1, 3, 3)
plt.plot(l_values, train_5b[1:])
plt.plot(l_values, test_5b[1:])
plt.title('train-100(1000)-100 vs. test-1000-100')
plt.legend(['train-100(1000)-100', 'test-1000-100'])
plt.xlabel('lamda')
plt.ylabel('MSE')

plt.subplots_adjust(left=.1, bottom=.1, right=.95,
                    top=0.95, wspace=.2, hspace=.4)
plt.show()
fig.savefig("Question1(b).png")
