import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import GRU, Dense, Input

import simpy
import tensorflow as tf

from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Load the dataset
file_path = '/Users/heeyaamin/PycharmProjects/ECC/dataset/dataset.txt'
data = pd.read_csv(file_path, header=None)
cpu_usage = data.values.flatten()

sequence_length = 10


def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


X, y = create_sequences(cpu_usage, sequence_length)
X_flattened = X.reshape(-1, sequence_length)
y_flattened = y

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_flattened)
y_scaled = scaler_y.fit_transform(y_flattened.reshape(-1, 1)).flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, train_size=0.7, shuffle=False)

# Define and train the Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predict
lr_predictions = model_lr.predict(X_test)

# Rescale predictions
lr_predictions_inv = scaler_y.inverse_transform(lr_predictions.reshape(-1, 1)).flatten()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Evaluate the model
mse_lr = mean_squared_error(y_test_inv, lr_predictions_inv)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test_inv, lr_predictions_inv)

print(f'Linear Regression Model - MSE: {mse_lr:.2f}, RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}')


# SimPy Datacenter Simulation

class Datacenter:
    def __init__(self, env, capacity=10):
        self.env = env
        self.server = simpy.Resource(env, capacity=capacity)

    def process_vm_request(self, vm, duration):
        with self.server.request() as req:
            yield self.env.timeout(duration)


class VMAllocator:
    def __init__(self, env, threshold=80):
        self.env = env
        self.threshold = threshold
        self.cumulative_cpu_usage = 0
        self.vm_counter = 0

    def allocate_vms(self, predicted_value, actual_value):
        self.cumulative_cpu_usage += predicted_value
        difference = abs(predicted_value - actual_value)

        if difference > 10:
            print(
                f"Significant prediction error: Predicted {predicted_value}%, Actual {actual_value}%, Difference {difference}%.")

        if self.cumulative_cpu_usage > self.threshold:
            num_vms = int((self.cumulative_cpu_usage - self.threshold) // 10) + 1
            print(f"Allocating {num_vms} VM(s) due to high cumulative usage ({self.cumulative_cpu_usage}%).")
            self.cumulative_cpu_usage = 0


env = simpy.Environment()

datacenter = Datacenter(env)
vm_allocator = VMAllocator(env)

last_n_values = X_scaled[-1]
predicted_values = []
actual_values = y_test_inv[:10]

for i, actual_value in enumerate(actual_values):
    predicted_value = scaler_y.inverse_transform(
        model_lr.predict([last_n_values]).reshape(-1, 1)
    ).flatten()[0]
    predicted_values.append(predicted_value)

    vm_allocator.allocate_vms(predicted_value, actual_value)

    last_n_values = np.append(last_n_values[1:], scaler_y.transform([[predicted_value]])).flatten()

    vm = f"VM_{i}_Predicted_{predicted_value:.2f}"
    env.process(datacenter.process_vm_request(vm, 10))

# Run the simulation
env.run(until=50)

# Load the dataset
file_path = '/Users/heeyaamin/PycharmProjects/ECC/dataset/dataset.txt'
data = pd.read_csv(file_path, header=None)
cpu_usage = data.values.flatten()

sequence_length = 10


def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


X, y = create_sequences(cpu_usage, sequence_length)
X_flattened = X.reshape(-1, sequence_length)
y_flattened = y

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_flattened)
y_scaled = scaler_y.fit_transform(y_flattened.reshape(-1, 1)).flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, train_size=0.7, shuffle=False)

# Define and train the GRU model
gru_model = Sequential()
gru_model.add(Input(shape=(sequence_length, 1)))  # Input layer
gru_model.add(GRU(50, activation='relu'))
gru_model.add(Dense(1))

gru_model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape data for GRU input
X_train_gru = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_gru = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the GRU model
gru_model.fit(X_train_gru, y_train, epochs=20, batch_size=32, verbose=0)

# Predict using GRU
gru_predictions = gru_model.predict(X_test_gru)

# Apply Linear Regression to the GRU predictions
model_lr = LinearRegression()
model_lr.fit(gru_predictions, y_test)

# Predict with Linear Regression on GRU predictions
lr_predictions = model_lr.predict(gru_predictions)

# Rescale predictions
lr_predictions_inv = scaler_y.inverse_transform(lr_predictions.reshape(-1, 1)).flatten()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Evaluate the model
mse_lr = mean_squared_error(y_test_inv, lr_predictions_inv)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test_inv, lr_predictions_inv)

print(f'Linear Regression Model - MSE: {mse_lr:.2f}, RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}')


# SimPy Datacenter Simulation

class Datacenter:
    def __init__(self, env, capacity=10):
        self.env = env
        self.server = simpy.Resource(env, capacity=capacity)

    def process_vm_request(self, vm, duration):
        with self.server.request() as req:
            yield self.env.timeout(duration)


class VMAllocator:
    def __init__(self, env, threshold=80):
        self.env = env
        self.threshold = threshold
        self.cumulative_cpu_usage = 0
        self.vm_counter = 0

    def allocate_vms(self, predicted_value, actual_value):
        self.cumulative_cpu_usage += predicted_value
        difference = abs(predicted_value - actual_value)

        if difference > 10:
            print(
                f"Significant prediction error: Predicted {predicted_value}%, Actual {actual_value}%, Difference {difference}%.")

        if self.cumulative_cpu_usage > self.threshold:
            num_vms = int((self.cumulative_cpu_usage - self.threshold) // 10) + 1
            print(f"Allocating {num_vms} VM(s) due to high cumulative usage ({self.cumulative_cpu_usage}%).")
            self.cumulative_cpu_usage = 0

        # Create SimPy environment


env = simpy.Environment()

# Create Datacenter
datacenter = Datacenter(env)

# Create VM Allocator
vm_allocator = VMAllocator(env)

# Initialize the last_n_values from the training set
last_n_values = X_scaled[-1]  # Get the last sequence from the dataset
predicted_values = []
actual_values = y_test_inv[:10]  # Get the first 10 test values

for i, actual_value in enumerate(actual_values):
    predicted_value = scaler_y.inverse_transform(
        model_lr.predict([gru_predictions[i]]).reshape(-1, 1)
    ).flatten()[0]
    predicted_values.append(predicted_value)

    # Allocate VMs based on the predicted value
    vm_allocator.allocate_vms(predicted_value, actual_value)

    last_n_values = np.append(last_n_values[1:], scaler_y.transform([[predicted_value]])).flatten()

    vm = f"VM_{i}_Predicted_{predicted_value:.2f}"
    env.process(datacenter.process_vm_request(vm, 10))

# Run the simulation
env.run(until=10)

# Load the dataset
file_path = '/Users/heeyaamin/PycharmProjects/ECC/dataset/dataset.txt'
data = pd.read_csv(file_path, header=None)
cpu_usage = data.values.flatten()

# Sequence length for LSTM
sequence_length = 10


# Create sequences for LSTM input
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


X, y = create_sequences(cpu_usage, sequence_length)

# Reshape data for LSTM (samples, timesteps, features)
X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_reshaped.reshape(-1, sequence_length))
X_scaled = X_scaled.reshape(-1, sequence_length, 1)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, train_size=0.7, shuffle=False)

# Define Bidirectional LSTM model
model_lstm = Sequential([
    Bidirectional(LSTM(50, return_sequences=False), input_shape=(sequence_length, 1)),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model_lstm.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Evaluate the model
lstm_predictions = model_lstm.predict(X_test)
lstm_predictions_inv = scaler_y.inverse_transform(lstm_predictions)

# Evaluate the model
mse_lstm = mean_squared_error(y_test, lstm_predictions_inv)
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(y_test, lstm_predictions_inv)

print(f'Bidirectional LSTM Model - MSE: {mse_lstm:.2f}, RMSE: {rmse_lstm:.2f}, R²: {r2_lstm:.2f}')


# SimPy Datacenter Simulation
class Datacenter:
    def __init__(self, env, capacity=10):
        self.env = env
        self.server = simpy.Resource(env, capacity=capacity)

    def process_vm_request(self, vm, duration):
        with self.server.request() as req:
            yield self.env.timeout(duration)


class VMAllocator:
    def __init__(self, env, threshold=80):
        self.env = env
        self.threshold = threshold
        self.cumulative_cpu_usage = 0
        self.vm_counter = 0

    def allocate_vms(self, predicted_value, actual_value):
        self.cumulative_cpu_usage += predicted_value
        difference = abs(predicted_value - actual_value)

        if difference > 10:
            print(
                f"Significant prediction error: Predicted {predicted_value}%, Actual {actual_value}%, Difference {difference}%.")

        if self.cumulative_cpu_usage > self.threshold:
            num_vms = int((self.cumulative_cpu_usage - self.threshold) // 10) + 1
            print(f"Allocating {num_vms} VM(s) due to high cumulative usage ({self.cumulative_cpu_usage}%).")
            self.cumulative_cpu_usage = 0


env = simpy.Environment()

datacenter = Datacenter(env)
vm_allocator = VMAllocator(env)

# Predict values and simulate VM allocation
last_n_values = X_scaled[-1]
predicted_values = []
actual_values = y_test[:10]

for i, actual_value in enumerate(actual_values):
    # Reshape the last_n_values for LSTM input (1, sequence_length, 1)
    last_n_values_reshaped = last_n_values.reshape(1, sequence_length, 1)

    # Predict the next value
    predicted_value = scaler_y.inverse_transform(
        model_lstm.predict(last_n_values_reshaped).reshape(-1, 1)
    ).flatten()[0]
    predicted_values.append(predicted_value)

    # Allocate VMs based on the predicted and actual values
    vm_allocator.allocate_vms(predicted_value, actual_value)

    # Update last_n_values for the next prediction
    last_n_values = np.append(last_n_values[1:], scaler_y.transform([[predicted_value]])).flatten()

    # Reshape last_n_values for the next LSTM prediction
    last_n_values_reshaped = last_n_values.reshape(1, sequence_length, 1)

    # Process VM request in the datacenter simulation
    vm = f"VM_{i}_Predicted_{predicted_value:.2f}"
    env.process(datacenter.process_vm_request(vm, 10))

# Run the simulation
env.run(until=10)
