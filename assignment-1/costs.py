import numpy as np

def log_cost(AL, Y):
    epsilon = 0.001
    cost = (1. / Y.shape[1]) * np.sum(-np.dot(Y, np.log(AL + epsilon).T) - np.dot(1 - Y, np.log(1 - AL + epsilon).T))
    return cost

def mape_cost(AL, Y):
    epsilon = 0.001
    cost = np.mean(np.abs((Y - AL) / (Y + epsilon))) * 100
    return cost

def mse_cost(AL, Y):
    cost = np.mean(np.square(AL - Y)) * 0.5
    return cost

def l2_regularization_cost(parameters, lambd, m):
    L = len(parameters) // 2
    sumw = 0
    for l in range(1, L + 1):
        sumw += np.sum(np.square(parameters['W' + str(l)]))
    L2_regularization_cost = (1 / m) * (lambd / 2) * (sumw)
    return L2_regularization_cost