'''
Arlind Stafaj
Question 1 code for Naive Bayes Algorithm
'''
import numpy as np
import pandas as pd
from collections import Counter

train = pd.read_csv('q1_train.csv')
train = train.drop(train.columns[0], axis=1)
attributes = list(train.columns)
train = train.sort_values([attributes[-1]])
lst = np.unique(np.array(train.iloc[:, -1]))
train = train.to_numpy()


def sub_array(data, var):
    arr = []
    for row in data:
        if row[-1] == var:
            arr.append(row)
    return np.array(arr)


def dict(data, ls):
    split = {}
    for a in ls:
        split[a] = sub_array(data, a)
    return split


split_train = dict(train, lst)

test = pd.read_csv('q1_test.csv')
test = test.drop(test.columns[0], axis=1)
test = test.to_numpy()


def get_result(a):
    result = []
    if a in split_train:
        probs = []
        for row in test:
            probs.append(np.count_nonzero(
                split_train[a] == a) / train.shape[0])
            for el in row:
                for i in range(train.T.shape[1]):
                    index = np.where(row == el)[0][0]
                    if index == i:
                        if el in split_train[a].T[i]:
                            ls = list(split_train[a].T[i])
                            probs.append(
                                (ls.count(el) + 1) / (len(ls) + len(Counter(split_train[a].T[i]).keys())))
                        else:
                            probs.append(
                                1/(1 + len(ls) + len(Counter(split_train[a].T[i]).keys())))
            result.append((np.prod(probs)))
            probs.clear()
    return result


res = {}

for a in lst:
    result = []
    result = get_result(a)
    res[a] = result

print('----------------------------------------------')
for i in range(len(res[lst[0]])):
    att = []
    val = []
    for a, v in res.items():
        att.append(a)
        val.append(v[i])
        print('  Instance ', i, '(', test[i], ')',
              ' for ', a, ' probability is', v[i] * 100)
    print('The predicted class label is :', att[val.index(max(val))])
    print('----------------------------------------------')
