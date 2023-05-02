import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import minimize, Bounds, LinearConstraint
from Trend_Following import * #Start, End, ret, dummy_L_df, months_between, next_month
from Utils import *

monthly_returns, asset_classes, asset = data_management(Start, End, '1mo')
print(monthly_returns)

'''
Here will be the optimization,

It needs to have monthly data fed into the optimization.

'''