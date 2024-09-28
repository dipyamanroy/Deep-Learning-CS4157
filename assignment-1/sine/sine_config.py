# Configuration for ANN

# Set hyperparameters
mini_batch_size = 64
learning_rate = 0.01
num_iterations = 10000
activation = "tanh" # or 'sigmoid', 'relu', etc.

# Regularization
regularisation = "L2"  # none or L2
lambda_reg = 0.1

# Optimizer
use_adam = True # Set to False if you want to use standard gradient descent
beta1 = 0.9
beta2 = 0.99
epsilon = 1e-8

# Define model
layers_dims = [1, 20, 20, 1]