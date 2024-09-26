# Configuration for ANN

# Set hyperparameters
mini_batch_size = 128
learning_rate = 0.001
num_iterations = 5000
activation = "tanh" # or 'sigmoid', 'relu', etc.

# Regularization
regularisation = "none"  # none or L2
lambda_reg = 0.1

# Optimizer
use_adam = False  # Set to False if you want to use standard gradient descent
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Define model
layers_dims = [1, 20, 20, 1]