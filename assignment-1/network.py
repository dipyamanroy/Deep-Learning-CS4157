# Imports
import random
import numpy as np
from activation import tanh, tanh_prime

class Networks(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Xavier initialization for weights
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # a is the input to the network
        for b, w in zip(self.biases, self.weights):
            a = tanh(np.dot(w, a) + b)  # Changed to tanh activation function
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        n = len(training_data)
        
        training_loss = []
        testing_loss = []
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            # Calculate training loss (MSE)
            train_loss = self.calculate_mse(training_data)
            training_loss.append(train_loss)
            
            # Always print training loss
            if test_data:
                test_loss = self.calculate_mse(test_data)
                testing_loss.append(test_loss)
                print(f"Epoch {j+1}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}")
            else:
                print(f"Epoch {j+1}: Train Loss {train_loss:.4f}")
        
        return training_loss, testing_loss


    def calculate_mse(self, data):
        total_loss = 0
        for x, y in data:
            prediction = self.feedforward(x)
            total_loss += np.mean((prediction - y) ** 2)
        return total_loss / len(data)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Forward pass
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = tanh(z)  # Changed to tanh activation
            activations.append(activation)
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * tanh_prime(zs[-1])  # Changed to tanh_prime
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = tanh_prime(z)  # Changed to tanh_prime
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)