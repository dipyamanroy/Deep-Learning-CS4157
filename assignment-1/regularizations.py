import numpy as np

# L2 Regularization
def l2_regularization(weights, lambda_reg):
    return 0.5 * lambda_reg * np.sum(np.square(weights))

def l2_regularization_gradient(weights, lambda_reg):
    return lambda_reg * weights
