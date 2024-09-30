import sys
import os
import json

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from network import Network

# Load the configuration from the JSON file
with open('config.json') as f:
    config = json.load(f)

# Read dataset
df = pd.read_csv("ccpp.csv")

# Scale the data between -0.9 and 0.9
df = ((df - df.min()) / (df.max() - df.min())) * (0.9 - (-0.9)) + (-0.9)

# Remove any nan values
df = df.dropna().reset_index(drop=True)

# Split data into train, val and test sets
# Last 10% is put in test set
test_size = round(df.shape[0] / 10)
test = df[-test_size:]

rem_size = (df.shape[0] - test.shape[0])
rem = df[:rem_size]

train = rem[rem.index % 5 != 0]  # Excludes every 5th row
val = rem[rem.index % 5 == 0]  # Selects every 5th

# Separate inputs and outputs
x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values
x_val = val.iloc[:, :-1].values
y_val = val.iloc[:, -1].values

# Putting the data in the right shape
x_train = np.transpose(x_train)
y_train = np.transpose(y_train)
y_train = y_train.reshape(1, -1)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)
y_test = y_test.reshape(1, -1)
x_val = np.transpose(x_val)
y_val = np.transpose(y_val)
y_val = y_val.reshape(1, -1)

print("No. of features: ", x_train.shape[0])
print("No. of samples in training set: ", x_train.shape[1])
print("No. of samples in testing set: ", x_test.shape[1])
print("No. of samples in validation set: ", x_val.shape[1])

# Create the network instance
network = Network(
    layers_dims=config['layers_dims'],
    optimizer=config['optimizer'],
    learning_rate=config['learning_rate'],
    he_init=True,
    mini_batch_size=config['mini_batch_size'],
    beta=config['beta'],
    beta1=config['beta1'],
    beta2=config['beta2'],
    epsilon=config['epsilon'],
    activation=config['activation'],
    regularisation=config['regularisation'],
    lambd=config['lambd'],
    cost_func=config['cost_func'],
    dropout_keep_prob=config['dropout_keep_prob']
)

# Train the model
network.train(
    X=x_train,
    Y=y_train,
    valid=True,
    valid_x=x_val,
    valid_y=y_val,
    num_iterations=config['num_iterations'],
    print_cost=True,
    early_stopping_patience=config['early_stopping_patience']
)

# MAPE
pred_val = network.predictvals(x_val)
mape_val = network.mape_cost(y_val, pred_val)
print("Validation MAPE : ", mape_val)

pred_test = network.predictvals(x_test)
mape_test = network.mape_cost(y_test, pred_test)
print("Test MAPE : ", mape_test)

# Plot R² scatter plot for validation set
network.plot_r2_scatter(x_val, y_val)