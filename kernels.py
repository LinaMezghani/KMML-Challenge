import numpy as np

def linear(x,y):
    return np.dot(x.T, y)

def quadratic(x,y):
    return np.dot(x, y.T) ** 2

def rbf(x,y):
    gamma = 0.01
    return np.exp(-gamma* (np.linalg.norm(x-y)**2))
