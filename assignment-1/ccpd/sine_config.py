# Configuration for ANN

# Set hyperparameters
mini_batch_size = 64
learning_rate = 0.01
num_iterations = 1000
activation = "tanh" # or 'sigmoid', 'relu', etc.
cost_func = "mse"

# Regularization
regularisation = "L2"  # none or L2
lambda_reg = 0.1

# Optimizer
optimizer = "adam"  # none or momentum or adam
beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Define model
layers_dims = [1, 20, 20, 1]

# Early stopping patience (optional, None for no early stopping)
early_stopping_patience = 0 # Stop if no improvement in validation error after 20 epochs (set to 0 to disable)