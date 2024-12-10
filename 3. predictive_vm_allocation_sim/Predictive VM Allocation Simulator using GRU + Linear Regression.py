import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import GRU, Dense, Input
import simpy

# Load the dataset
file_path = '/dataset/dataset.txt'
data = pd.read_csv(file_path, header=None)
cpu_usage = data.values.flatten()

# Define parameters
sequence_length = 10


def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


X, y = create_sequences(cpu_usage, sequence_length)

# Flatten X for Linear Regression
X_flattened = X.reshape(-1, sequence_length)
y_flattened = y

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_flattened)
y_scaled = scaler_y.fit_transform(y_flattened.reshape(-1, 1)).flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, train_size=0.8, shuffle=False)

# Define and train the GRU model
gru_model = Sequential()
gru_model.add(Input(shape=(sequence_length, 1)))  # Input layer
gru_model.add(GRU(50, activation='relu'))
gru_model.add(Dense(1))

gru_model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape data for GRU input
X_train_gru = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_gru = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the GRU model with reduced epochs for faster training
gru_model.fit(X_train_gru, y_train, epochs=20, batch_size=32, verbose=0)

# Predict using GRU
gru_predictions = gru_model.predict(X_test_gru)

# Apply Linear Regression to the GRU predictions
model_lr = LinearRegression()
model_lr.fit(gru_predictions, y_test)

# Predict with Linear Regression on GRU predictions
lr_predictions = model_lr.predict(gru_predictions)

# Rescale predictions
lr_predictions_inv = scaler_y.inverse_transform(lr_predictions.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
mse_lr = mean_squared_error(y_test_inv, lr_predictions_inv)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test_inv, lr_predictions_inv)

print(f'Linear Regression Model - MSE: {mse_lr}, RMSE: {rmse_lr}, R²: {r2_lr}')


# Function to predict CPU usage and return a range (20% range as an example)
def predict_cpu_usage(last_n_values):
    # Predict using GRU
    gru_input = np.array(last_n_values).reshape(1, sequence_length, 1)
    gru_pred = gru_model.predict(gru_input)

    # Adjust prediction with Linear Regression
    lr_pred = model_lr.predict(gru_pred)

    # Predict range with 20% variation
    predicted_range = (lr_pred - (0.2 * lr_pred), lr_pred + (0.2 * lr_pred))  # 20% range
    return lr_pred[0] * 100, predicted_range  # Multiply by 100 to scale the result


# SimPy Datacenter Simulation

class Datacenter:
    def __init__(self, env, capacity=10):
        self.env = env
        self.server = simpy.Resource(env, capacity=capacity)

    def process_vm_request(self, vm):
        with self.server.request() as req:
            yield req
            print(f"{vm} starts at {self.env.now}")
            yield self.env.timeout(10)  # Simulate VM processing time
            print(f"{vm} finishes at {self.env.now}")


class VMAllocator:
    def __init__(self, env, threshold=80):
        self.env = env
        self.threshold = threshold
        self.cumulative_cpu_usage = 0  # Track the cumulative CPU usage
        self.vm_counter = 0  # Counter to assign unique VM numbers

    def allocate_vms(self, predicted_value, predicted_range):
        self.cumulative_cpu_usage += predicted_value  # Add the predicted value to cumulative usage
        print(f"Cumulative CPU usage: {self.cumulative_cpu_usage}%")

        # Check if cumulative CPU usage exceeds the threshold
        if self.cumulative_cpu_usage > self.threshold:
            # If predicted CPU usage is high, prepare more VMs
            num_vms = int((self.cumulative_cpu_usage - self.threshold) // 10) + 1
            print(f"Allocating {num_vms} VMs due to cumulative CPU usage {self.cumulative_cpu_usage}%")
            for i in range(num_vms):
                print(f"VM{self.vm_counter} VM_for_{predicted_value} starts at 0")
                self.vm_counter += 1
            self.cumulative_cpu_usage = 0  # Reset cumulative CPU usage after allocation
        else:
            print(f"CPU usage predicted to be {predicted_value}%. No additional VMs needed.")


# Create SimPy environment
env = simpy.Environment()

# Create Datacenter
datacenter = Datacenter(env)

# Create VM Allocator
vm_allocator = VMAllocator(env)

# Predict the next 10 values and allocate VMs based on the predicted CPU usage
last_n_values = X_scaled[-1]  # Get the last sequence from the dataset
predicted_values = []

for _ in range(10):
    predicted_value, predicted_range = predict_cpu_usage(last_n_values)
    predicted_values.append((predicted_value, predicted_range))

    # Update the last_n_values to include the latest predicted value
    last_n_values = np.append(last_n_values[1:],
                              predicted_value / 100)  # Scale back to 0-1 range for the next prediction

    # Allocate VMs based on the predicted value
    vm_allocator.allocate_vms(predicted_value, predicted_range)

# Simulate VM processing based on predicted values
for predicted_value, predicted_range in predicted_values:
    vm = f"VM_for_{predicted_value}"  # Create VM name based on predicted value
    env.process(datacenter.process_vm_request(vm))

# Run the simulation
env.run(until=20)