import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import simpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Load the dataset
file_path = '/Users/heeyaamin/PycharmProjects/ECC/dataset/dataset.txt'
data = pd.read_csv(file_path, header=None)
cpu_usage = data.values.flatten()

# Define parameters
sequence_length = 10


# Prepare the data for the Bi-LSTM model (previous n values predict next value)
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


X, y = create_sequences(cpu_usage, sequence_length)

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, train_size=0.8, shuffle=False)

# Reshape data for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build and train Bidirectional LSTM model
model_bi_lstm = Sequential()
model_bi_lstm.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model_bi_lstm.add(Bidirectional(LSTM(units=50)))
model_bi_lstm.add(Dense(1))
model_bi_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_bi_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict using the Bi-LSTM model
bi_lstm_predictions = model_bi_lstm.predict(X_test)
bi_lstm_predictions = scaler_y.inverse_transform(bi_lstm_predictions)
y_test_inv_bi = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mse_bi_lstm = mean_squared_error(y_test_inv_bi, bi_lstm_predictions)
rmse_bi_lstm = np.sqrt(mse_bi_lstm)
r2_bi_lstm = r2_score(y_test_inv_bi, bi_lstm_predictions)

print(f'Bidirectional LSTM - MSE: {mse_bi_lstm}, RMSE: {rmse_bi_lstm}, R²: {r2_bi_lstm}')


# Function to predict CPU usage and return a range (20% range as an example)
def predict_cpu_usage(last_n_values):
    prediction = model_bi_lstm.predict(last_n_values.reshape(1, -1, 1))
    prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1))
    predicted_range = (prediction - (0.2 * prediction), prediction + (0.2 * prediction))  # 20% range
    return prediction[0][0], predicted_range


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
            print(f"Allocating {num_vms + 1} VMs due to cumulative CPU usage {self.cumulative_cpu_usage}%")
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