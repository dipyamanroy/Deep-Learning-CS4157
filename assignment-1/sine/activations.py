import numpy as np

# Activation functions
def tanh(Z):
    """Tanh activation function"""
    return np.tanh(Z)

def tanh_prime(Z):
    """Derivative of the tanh function"""
    return 1.0 - np.tanh(Z) ** 2

def sigmoid(Z):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_prime(Z):
    """Derivative of the sigmoid function"""
    return sigmoid(Z) * (1 - sigmoid(Z))

def relu(Z):
    """ReLU (Rectified Linear Unit) activation function"""
    return np.maximum(0, Z)

def relu_prime(Z):
    """Derivative of the ReLU function"""
    return np.where(Z > 0, 1.0, 0.0)