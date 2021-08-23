'''
    Arlind Stafaj HW1
    Question 2
'''
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import *
from functions import *

cv_results('train-100-10.xlsx', 'test-100-10.xlsx')

cv_results("train-100-100.xlsx", "test-100-100.xlsx")

cv_results("train-1000-100.xlsx", "test-1000-100.xlsx")

cv_results("train-50(1000)-100.xlsx", "test-1000-100.xlsx")

cv_results("train-100(1000)-100.xlsx", "test-1000-100.xlsx")

cv_results("train-150(1000)-100.xlsx", "test-1000-100.xlsx")
