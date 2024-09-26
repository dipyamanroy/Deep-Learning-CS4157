import random
import numpy as np
from activations import tanh, tanh_prime, sigmoid, sigmoid_prime, relu, relu_prime

class Networks(object):
    def __init__(self, sizes, activation='tanh', use_adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8, lambda_reg=0.0):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Xavier initialization for weights
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

        # Adam optimizer variables
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize Adam parameters if using Adam
        if use_adam:
            self.m_w = [np.zeros(w.shape) for w in self.weights]
            self.v_w = [np.zeros(w.shape) for w in self.weights]
            self.m_b = [np.zeros(b.shape) for b in self.biases]
            self.v_b = [np.zeros(b.shape) for b in self.biases]
            self.t = 0  # time step

        self.lambda_reg = lambda_reg  # L2 regularization factor
        
        # Set the activation function and its derivative
        self.activation, self.activation_prime = self._select_activation(activation)

    def _select_activation(self, activation):
        """Selects the activation function and its derivative based on the input."""
        if activation == 'tanh':
            return tanh, tanh_prime
        elif activation == 'sigmoid':
            return sigmoid, sigmoid_prime
        elif activation == 'relu':
            return relu, relu_prime
        else:
            raise ValueError("Unsupported activation function")

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation(np.dot(w, a) + b)
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

        # Add L2 regularization term
        if self.lambda_reg > 0:
            l2_penalty = 0.5 * self.lambda_reg * sum(np.linalg.norm(w) ** 2 for w in self.weights)
            total_loss += l2_penalty

        return total_loss / len(data)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Apply L2 regularization to weights
        if self.lambda_reg > 0:
            nabla_w = [nw + self.lambda_reg * w for nw, w in zip(nabla_w, self.weights)]

        if self.use_adam:
            self.t += 1
            # Update Adam parameters
            self.m_w = [self.beta1 * mw + (1 - self.beta1) * nw for mw, nw in zip(self.m_w, nabla_w)]
            self.v_w = [self.beta2 * vw + (1 - self.beta2) * (nw ** 2) for vw, nw in zip(self.v_w, nabla_w)]
            self.m_b = [self.beta1 * mb + (1 - self.beta1) * nb for mb, nb in zip(self.m_b, nabla_b)]
            self.v_b = [self.beta2 * vb + (1 - self.beta2) * (nb ** 2) for vb, nb in zip(self.v_b, nabla_b)]

            # Compute bias-corrected first and second moment estimates
            m_w_hat = [mw / (1 - self.beta1 ** self.t) for mw in self.m_w]
            v_w_hat = [vw / (1 - self.beta2 ** self.t) for vw in self.v_w]
            m_b_hat = [mb / (1 - self.beta1 ** self.t) for mb in self.m_b]
            v_b_hat = [vb / (1 - self.beta2 ** self.t) for vb in self.v_b]

            # Update weights and biases using Adam
            self.weights = [w - (eta * mw_hat) / (np.sqrt(vw_hat) + self.epsilon) for w, mw_hat, vw_hat in zip(self.weights, m_w_hat, v_w_hat)]
            self.biases = [b - (eta * mb_hat) / (np.sqrt(vb_hat) + self.epsilon) for b, mb_hat, vb_hat in zip(self.biases, m_b_hat, v_b_hat)]
        else:
            # Standard gradient descent update
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
            activation = self.activation(z)  # Changed to dynamic activation function
            activations.append(activation)
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_prime(z)  # Changed to dynamic activation prime
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)