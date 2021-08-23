'''
Arlind Stafaj 
Data Mining 6930 HW4
Question 3 parts a, b, c, e
'''
import math


def combination(n: int, r: int):
    return math.factorial(n)/(math.factorial(n-r) * math.factorial(r))


def majority_vote(models, p):
    kCon = models/2
    n = models
    k = 0
    result = 0
    while(k < n/2):
        result += combination(n, k) * pow(p, k) * pow((1-p), (n-k))
        k += 1
    return result


print('Question 3 (a): ', majority_vote(3, 0.6)*100)
print('Question 3 (b): ', majority_vote(5, 0.6)*100)
print('Question 3 (c): ', majority_vote(25, 0.6)*100)
print('Question 3 (e): ', majority_vote(25, 0.45)*100)
