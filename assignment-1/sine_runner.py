import numpy as np
import matplotlib.pyplot as plt
from network import Networks
from sine_config import (mini_batch_size, learning_rate, num_iterations, 
                    layers_dims, lambda_reg, use_adam, beta1, beta2, epsilon, 
                    activation)

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
net = Networks(sizes=layers_dims, 
                activation=activation,  
                beta1=beta1, 
                beta2=beta2, 
                epsilon=epsilon, 
                lambda_reg=lambda_reg)

# Split the training data into training and testing sets
training_data = generate_sine_data(samples=800)  # 800 samples for training
testing_data = generate_sine_data(samples=200)   # 200 samples for testing

# Train the network
training_loss, testing_loss = net.SGD(training_data, epochs=num_iterations, 
                                        mini_batch_size=mini_batch_size, 
                                        eta=learning_rate, 
                                        test_data=testing_data)

# Testing the network on new data
test_x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
test_y = np.sin(test_x)
predicted_y = [net.feedforward(np.reshape(i, (1, 1))) for i in test_x]
predicted_y = np.array(predicted_y).flatten()

# Ensure predictions are numeric
predicted_y = np.clip(predicted_y, -10, 10)  # Clip large values

# Calculate R² for the testing set
r2 = calculate_r2(test_y, predicted_y)
print(f"Test R²: {r2:.4f}")

# Use LaTeX for fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# Plotting the actual vs predicted sine wave
plt.figure(figsize=(10, 6))
plt.plot(test_x, test_y, label=r'\textbf{Actual sine}', color='darkorange', linewidth=2)
plt.plot(test_x, predicted_y, label=r'\textbf{Predicted sine}', linestyle='dashed', color='dodgerblue', linewidth=2)

plt.title(r'\textbf{Predicted vs actual sine wave}', fontsize=14)
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$\sin(x)$', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.2)
plt.legend(loc='best', fontsize=12)
plt.savefig("sine_prediction.png", dpi=300, bbox_inches='tight')

# Convergence plot for training and testing loss
plt.figure(figsize=(10, 6))
plt.plot(training_loss, label=r'\textbf{Training Loss (MSE)}', color='darkorange', linewidth=2)
plt.plot(testing_loss, label=r'\textbf{Testing Loss (MSE)}', color='dodgerblue', linewidth=1)

plt.title(r'$\textbf{Convergence History}$', fontsize=14)
plt.xlabel(r'$\textbf{Epochs}$', fontsize=12)
plt.ylabel(r'$\textbf{Loss (MSE)}$', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.2)
plt.legend(loc='best', fontsize=12)
plt.savefig("loss_convergence.png", dpi=300, bbox_inches='tight')

# Scatter plot for R²
plt.figure(figsize=(10, 6))
plt.scatter(test_y, predicted_y, color='dodgerblue', alpha=0.7)
plt.plot([-1, 1], [-1, 1], color='darkorange', linestyle='dashed', linewidth=1)  # Line for perfect predictions

plt.title(r'\textbf{R²: Predicted vs True Values}', fontsize=14)
plt.xlabel(r'$\textbf{True Values}$', fontsize=12)
plt.ylabel(r'$\textbf{Predicted Values}$', fontsize=12)

# Display R² score on the plot
plt.annotate(f'R²: {r2:.4f}', xy=(0.5, 0.1), xycoords='axes fraction', fontsize=12, ha='center', color='black', bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='lightgray'))

plt.grid(True, linestyle='--', alpha=0.2)
plt.savefig("r2_scatter_plot.png", dpi=300, bbox_inches='tight')