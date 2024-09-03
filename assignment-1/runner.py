import numpy as np
import matplotlib.pyplot as plt

from network import Networks

# Generate training data
def generate_sine_data(samples=1000):
    x = np.linspace(0, 2 * np.pi, samples)
    y = np.sin(x)
    training_data = [(np.reshape(i, (1, 1)), np.reshape(j, (1, 1))) for i, j in zip(x, y)]
    return training_data

# 1 input neuron, 3 hidden neurons, and 1 output neuron
net = Networks([1, 5, 5, 1])

training_data = generate_sine_data()

net.SGD(training_data, epochs=1000, mini_batch_size=20, eta=0.2)

# Testing the network on new data
test_x = np.linspace(0, 2 * np.pi, 100)
test_y = np.sin(test_x)
predicted_y = [net.feedforward(np.reshape(i, (1, 1))) for i in test_x]
predicted_y = np.array(predicted_y).flatten()

# Plot the results
plt.plot(test_x, test_y, label="Actual Sine")
plt.plot(test_x, predicted_y, label="Predicted Sine", linestyle='dashed')
plt.legend()
plt.savefig("sine_prediction.png")