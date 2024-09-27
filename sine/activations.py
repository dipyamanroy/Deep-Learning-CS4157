import numpy as np

# Activation functions
def tanh(z):
    """Tanh activation function"""
    return np.tanh(z)

def tanh_prime(z):
    """Derivative of the tanh function"""
    return 1.0 - np.tanh(z) ** 2

def sigmoid(z):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    """ReLU (Rectified Linear Unit) activation function"""
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of the ReLU function"""
    return np.where(z > 0, 1.0, 0.0)