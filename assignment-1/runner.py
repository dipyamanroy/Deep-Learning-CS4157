import numpy as np
import matplotlib.pyplot as plt
from network import Networks
from config import (mini_batch_size, learning_rate, num_iterations, 
                    layers_dims, lambda_reg, use_adam, beta1, beta2, epsilon, 
                    activation)

# Generate training data
def generate_sine_data(samples=1000):
    x = np.linspace(0, 2 * np.pi, samples)
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
test_x = np.linspace(0, 2 * np.pi, 100)
test_y = np.sin(test_x)
predicted_y = [net.feedforward(np.reshape(i, (1, 1))) for i in test_x]
predicted_y = np.array(predicted_y).flatten()

# Use LaTeX for fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# Plotting the actual vs predicted sine wave
plt.figure(figsize=(10, 6))
plt.plot(test_x, test_y, label=r'\textbf{Actual Sine}', color='darkorange', linewidth=2)
plt.plot(test_x, predicted_y, label=r'\textbf{Predicted Sine}', linestyle='dashed', color='dodgerblue', linewidth=2)

plt.title(r'\textbf{Prediction vs Actual Sine Wave}', fontsize=14)
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$\sin(x)$', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.2)
plt.legend(loc='best', fontsize=12)
plt.savefig("sine_prediction.png", dpi=300, bbox_inches='tight')

# Convergence plot for training and testing loss
plt.figure(figsize=(10, 6))
plt.plot(training_loss, label=r'\textbf{Training Loss (MSE)}', color='darkorange', linewidth=2)
plt.plot(testing_loss, label=r'\textbf{Testing Loss (MSE)}', color='dodgerblue', linewidth=1)

plt.title(r'$\textbf{Convergence of Training and Testing Loss}$', fontsize=14)
plt.xlabel(r'$\textbf{Epochs}$', fontsize=12)
plt.ylabel(r'$\textbf{Loss (MSE)}$', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.2)
plt.legend(loc='best', fontsize=12)
plt.savefig("loss_convergence.png", dpi=300, bbox_inches='tight')