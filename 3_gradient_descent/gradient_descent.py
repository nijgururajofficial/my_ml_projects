import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002
    cost_prev = 0
    
    for i in range(iterations):
        y_pred = m_curr * x + b_curr
        cost = (1/n)*sum([value**2 for value in (y-y_pred)])
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_prev,rel_tol=1e-20):
            break
        cost_prev = cost
    return m_curr, b_curr

def linear_reg():
    df = pd.read_csv("test_scores.csv")
    reg = linear_model.LinearRegression()
    reg.fit(df[['math']],df.cs)
    return reg.coef_, reg.intercept_

if __name__ == '__main__':
    df = pd.read_csv("test_scores.csv")
    x = df['math'].to_numpy()
    y = df['cs'].to_numpy()
    
    m,b = gradient_descent(x,y)
    print(f"Using gradient descent m {m} b {b}")
    
    m_reg,b_reg = linear_reg()
    print(f"Using sklearn m {m_reg} b {b_reg}")
    
