import numpy as np
import matplotlib.pyplot as plt
import math
from activations import *
from costs import *

class Network:
    def __init__(self, layers_dims, optimizer='none', learning_rate=0.0007, he_init=False, 
                    mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                    activation='sigmoid', regularisation='none', lambd=0.1, cost_func='mse', dropout_keep_prob=1.0):
        self.layers_dims = layers_dims
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.he_init = he_init
        self.mini_batch_size = mini_batch_size
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.activation = activation
        self.regularisation = regularisation
        self.lambd = lambd
        self.cost_func = cost_func
        self.dropout_keep_prob = dropout_keep_prob
        self.parameters = None
        self.costs = []
        self.validcosts = []

    def initialize_parameters(self, n_x, n_h, n_y):
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        
        assert(W1.shape == (n_h, n_x))
        assert(b1.shape == (n_h, 1))
        assert(W2.shape == (n_y, n_h))
        assert(b2.shape == (n_y, 1))
        
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def initialize_parameters_deep(self, layer_dims):
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    def initialize_parameters_deep_he(self, layers_dims):
        parameters = {}
        L = len(layers_dims) - 1
        
        for l in range(1, L + 1):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            
        return parameters

    def linear_forward(self, A, W, b):
        Z = W.dot(A) + b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
            cache = (linear_cache, activation_cache)
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < self.dropout_keep_prob
            A = np.multiply(A, D)
            A = A / self.dropout_keep_prob
            cache = (linear_cache, activation_cache, D)
        elif activation == "tanh":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = tanh(Z)
            cache = (linear_cache, activation_cache)
        else:
            raise ValueError("Invalid activation function. Supported activations are 'sigmoid', 'relu', and 'tanh'.")

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        return A, cache

    def L_model_forward(self, X, parameters, activation):
        caches = []
        A = X
        L = len(parameters) // 2
        
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
            caches.append(cache)
        
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation)
        caches.append(cache)
        
        assert(AL.shape == (1, X.shape[1]))
        return AL, caches

    def compute_cost(self, AL, Y, parameters, lambd, regularisation, cost_func='mse'):
        m = Y.shape[1]
        L = len(parameters) // 2
        epsilon = 0.001
        
        if cost_func == 'log':
            cost = log_cost(AL, Y)
        elif cost_func == 'mape':
            cost = mape_cost(AL, Y)
        elif cost_func == 'mse':
            cost = mse_cost(AL, Y)

        if regularisation == 'L2':
            cost += l2_regularization_cost(parameters, lambd)

        cost = np.squeeze(cost)
        assert(cost.shape == ())
        return cost

    def linear_backward(self, dZ, cache, regularisation, lambd):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1. / m * np.dot(dZ, A_prev.T)
        if regularisation == 'L2':
            dW += lambd * W * (1 / m)

        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, regularisation, lambd, activation):
        if activation == "relu":
            linear_cache, activation_cache, D = cache
            dZ = relu_backward(dA, activation_cache)
            dZ = np.multiply(dZ, D)
            dZ = dZ / self.dropout_keep_prob
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, regularisation, lambd)
        elif activation == "sigmoid":
            linear_cache, activation_cache = cache
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, regularisation, lambd)
        elif activation == "tanh":
            linear_cache, activation_cache = cache
            dZ = tanh_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, regularisation, lambd)
        
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches, activation, regularisation, lambd, cost_func='log'):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        if cost_func == 'log':
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        elif cost_func == 'mse':
            dAL = (AL - Y)
        
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, regularisation, lambd, activation)
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, regularisation, lambd, activation="relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        L = len(parameters) // 2
        
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
        return parameters

    def initialize_adam(self, parameters):
        L = len(parameters) // 2
        v = {}
        s = {}
        
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
            s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
        return v, s

    def initialize_velocity(self, parameters):
        L = len(parameters) // 2
        v = {}
        
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
        return v

    def update_parameters_with_adam(self, parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon):
        L = len(parameters) // 2
        v_corrected = {}
        s_corrected = {}
        
        for l in range(L):
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - math.pow(beta1, t))
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - math.pow(beta1, t))
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads["dW" + str(l+1)] * grads["dW" + str(l+1)]
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l+1)] * grads["db" + str(l+1)]
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - math.pow(beta2, t))
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - math.pow(beta2, t))
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
        
        return parameters, v, s

    def update_parameters_with_momentum(self, parameters, grads, v, beta, learning_rate):
        L = len(parameters) // 2
        
        for l in range(L):
            v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * grads["db" + str(l+1)]
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
        
        return parameters, v

    def random_mini_batches(self, X, Y, mini_batch_size, seed):
        np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []
        
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        num_complete_minibatches = math.floor(m / mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : ]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : ]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches

    def train(self, X, Y, valid=False, valid_x=None, valid_y=None, num_iterations=10000, print_cost=True, early_stopping_patience=5):
        L = len(self.layers_dims)
        t = 0
        seed = 10
        m = X.shape[1]
        batches = m // self.mini_batch_size
        
        if self.he_init:
            self.parameters = self.initialize_parameters_deep_he(self.layers_dims)
        else:
            self.parameters = self.initialize_parameters_deep(self.layers_dims)

        # Initialize the optimizer
        if self.optimizer == "gd":
            pass  # no initialization required for gradient descent
        elif self.optimizer == "momentum":
            v = self.initialize_velocity(self.parameters)
        elif self.optimizer == "adam":
            v, s = self.initialize_adam(self.parameters)

        # Optimization loop
        for i in range(num_iterations):
            seed = seed + 1
            minibatches = self.random_mini_batches(X, Y, self.mini_batch_size, seed)
            cost_total = 0

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                a3, caches = self.L_model_forward(minibatch_X, self.parameters, self.activation)
                cost_total += self.compute_cost(a3, minibatch_Y, self.parameters, self.lambd, self.regularisation, self.cost_func)
                grads = self.L_model_backward(a3, minibatch_Y, caches, self.activation, self.regularisation, self.lambd, self.cost_func)

                if self.optimizer == "gd" or self.optimizer == 'none':
                    self.parameters = self.update_parameters(self.parameters, grads, self.learning_rate)
                elif self.optimizer == "momentum":
                    self.parameters, v = self.update_parameters_with_momentum(self.parameters, grads, v, self.beta, self.learning_rate)
                elif self.optimizer == "adam":
                    t = t + 1
                    self.parameters, v, s = self.update_parameters_with_adam(self.parameters, grads, v, s, t, self.learning_rate, self.beta1, self.beta2, self.epsilon)

            cost_avg = cost_total / batches

            if print_cost and i % 1 == 0:
                if valid:
                    valid_error = self.predicted_error(valid_x, valid_y)
                    self.validcosts.append(valid_error)
                    if early_stopping_patience > 0:
                        best_valid_error = float('inf')
                        patience = 0
                        if valid_error < best_valid_error:
                            best_valid_error = valid_error
                            patience = 0
                    else:
                        patience += 1
                    if patience >= early_stopping_patience:
                        print("Early stopping: validation error didn't improve for {} epochs".format(early_stopping_patience))
                        break
                    print("Cost after epoch %i: %f, Valid err: %f" % (i, cost_avg, valid_error))
                else:
                    print("Cost after epoch %i: %f" % (i, cost_avg))
                self.costs.append(cost_avg)

        # Convergence plot for training and validation costs
        plt.figure(figsize=(10, 6))

        # Plot training costs
        plt.plot(self.costs, label=r'\textbf{Training Cost}', color='darkorange', linewidth=2)

        # Plot validation costs if applicable
        if valid:
            plt.plot(self.validcosts, label=r'\textbf{Validation Cost}', color='dodgerblue', linewidth=1)

        plt.title(r'$\textbf{Convergence History}$', fontsize=14)
        plt.xlabel(r'$\textbf{Epochs}$', fontsize=12)
        plt.ylabel(r'$\textbf{Cost}$', fontsize=12)

        plt.grid(True, linestyle='--', alpha=0.2)
        plt.legend(loc='best', fontsize=12)

        plt.savefig("cost_convergence.png", dpi=300, bbox_inches='tight')

    def predict(self, X, y):
        m = X.shape[1]
        n = len(self.parameters) // 2
        p = np.zeros((1, m))

        probas, caches = self.L_model_forward(X, self.parameters, self.activation)

        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        print("Accuracy: " + str(np.sum((p == y) / m)))
        return p

    def predictvals(self, x):
        probas, caches = self.L_model_forward(x, self.parameters, self.activation)
        return probas

    def predicted_error(self, x, y):
        probas, caches = self.L_model_forward(x, self.parameters, self.activation)
        err = self.compute_cost(probas, y, self.parameters, self.lambd, self.regularisation, self.cost_func)
        return err

    def mape_cost(self, y, a):
        return mape_cost(y, a)
        
    # Use LaTeX for fonts
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']

    def plot_r2_scatter(self, X, y):
        predictions = self.predictvals(X)
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, color='dodgerblue', alpha=0.7)
        plt.plot([-0.5, 0.8], [-0.5, 0.8], color='darkorange', linestyle='dashed', linewidth=1)  # Line for perfect predictions
        
        plt.title(r'\textbf{R²: Predicted vs True Values}', fontsize=14)
        plt.xlabel(r'$\textbf{True Values}$', fontsize=12)
        plt.ylabel(r'$\textbf{Predicted Values}$', fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.savefig("r2_scatter_plot.png", dpi=300, bbox_inches='tight')