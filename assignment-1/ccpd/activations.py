import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0 # Correction
    assert (dZ.shape == Z.shape)
    return dZ

def tanh(Z):
    A=np.tanh(Z)
    cache =Z
    return A,cache

def tanh_backward(dA,cache):
    Z = cache
    s=np.tanh(Z)
    dZ=dA*(1-s*s)
    assert (dZ.shape == Z.shape)
    return dZ

def leaky_relu(Z):
    return np.maximum(0.01 * Z, Z)

def leaky_relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0.01
    return dZ

def elu(Z, alpha=1.0):
    return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))

def elu_backward(dA, Z, alpha=1.0):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = alpha * np.exp(Z[Z <= 0])
    return dZ