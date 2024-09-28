# Configuration for ANN

# Set hyperparameters
mini_batch_size = 256
learning_rate = 0.001
num_iterations = 500
activation = "tanh"
cost_func = "mse"

# Regularization
regularisation = "none"  # none or L2
lambd = 0.1

# Optimizer
optimizer = "adam"  # none or momentum or adam
beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Define model
layers_dims = [4, 10, 10, 1]

# Dropout
dropout_keep_prob = 0.8  # Set to 1.0 to disable dropout

# Early stopping patience (optional, None for no early stopping)
early_stopping_patience = 20  # Stop if no improvement in validation error after 20 epochs (set to None to disable)