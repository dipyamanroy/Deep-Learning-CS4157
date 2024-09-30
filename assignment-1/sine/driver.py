import sys
import os
import json

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from network import Network

# Load the configuration from the JSON file
with open('config.json') as f:
    config = json.load(f)

# Function to calculate R² score
def calculate_r2(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Generate training data
def generate_sine_data(samples=1000):
    x = np.linspace(-2 * np.pi, 2 * np.pi, samples)
    y = np.sin(x)
    training_data = [(np.reshape(i, (1, 1)), np.reshape(j, (1, 1))) for i, j in zip(x, y)]
    return training_data

# Initialize the network with the specified activation function
net = Network(layers_dims=config['layers_dims'], 
                activation=config['activation'],  
                beta1=config['beta1'], 
                beta2=config['beta2'], 
                epsilon=config['epsilon'], 
                lambd=config['lambda_reg'],
                cost_func=config['cost_func'],
                optimizer=config['optimizer'],
                learning_rate=config['learning_rate'],
                mini_batch_size=config['mini_batch_size'])

# Split the training data into training and testing sets
training_data = generate_sine_data(samples=800)  # 800 samples for training
testing_data = generate_sine_data(samples=200)   # 200 samples for testing

# Convert data to numpy arrays
X_train = np.array([x[0] for x in training_data]).reshape(1, 800)
Y_train = np.array([x[1] for x in training_data]).reshape(1, 800)
X_test = np.array([x[0] for x in testing_data]).reshape(1, 200)
Y_test = np.array([x[1] for x in testing_data]).reshape(1, 200)

# Train the network
net.train(X_train, Y_train, valid=True, valid_x=X_test, valid_y=Y_test, num_iterations=config['num_iterations'], early_stopping_patience=config['early_stopping_patience'])

# Testing the network on new data
test_x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
test_x = test_x.reshape(1, 100)
predicted_y = net.predictvals(test_x).flatten()

# Ensure predictions are numeric
predicted_y = np.clip(predicted_y, -10, 10)  # Clip large values

# Use LaTeX for fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# Plotting the actual vs predicted sine wave
plt.figure(figsize=(10, 6))
plt.plot(test_x.flatten(), np.sin(test_x.flatten()), label=r'\textbf{Actual sine}', color='darkorange', linewidth=2)
plt.plot(test_x.flatten(), predicted_y, label=r'\textbf{Predicted sine}', linestyle='dashed', color='dodgerblue', linewidth=2)

plt.title(r'\textbf{Predicted vs actual sine wave}', fontsize=14)
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$\sin(x)$', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.2)
plt.legend(loc='best', fontsize=12)
plt.savefig("sine_prediction.png", dpi=300, bbox_inches='tight')

# Now generate the R² scatter plot
net.plot_r2_scatter(test_x, np.sin(test_x.flatten()))