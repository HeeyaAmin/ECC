import simpy
import random

# Define a datacenter environment
class Datacenter:
    def __init__(self, env):
        self.env = env
        self.server = simpy.Resource(env, capacity=1)  # Server with single resource

    def process_vm_request(self, vm):
        with self.server.request() as req:
            yield req  # Wait for server availability
            yield self.env.timeout(vm.runtime)  # Simulate VM runtime

# Define a virtual machine
class VirtualMachine:
    def __init__(self, env, runtime):
        self.env = env
        self.runtime = runtime

# Define a simulation scenario
def simulate(env, datacenter):
    vm = VirtualMachine(env, runtime=10)  # Create a VM with 10 time units runtime
    yield env.process(datacenter.process_vm_request(vm))

# Create simulation environment and datacenter
env = simpy.Environment()
datacenter = Datacenter(env)

# Run simulation
env.process(simulate(env, datacenter))
env.run(until=20)  # Run simulation for 20 time units



class Datacenter:
    def __init__(self, env):
        self.env = env
        self.server = simpy.Resource(env, capacity=1)

    def process_vm_request(self, vm):
        with self.server.request() as req:
            yield req
            print(f"Virtual machine starts running at {self.env.now}")
            yield self.env.timeout(vm.runtime)
            print(f"Virtual machine finishes at {self.env.now}")

class VirtualMachine:
    def __init__(self, env, runtime):
        self.env = env
        self.runtime = runtime

def simulate(env, datacenter):
    vm = VirtualMachine(env, runtime=10)
    yield env.process(datacenter.process_vm_request(vm))

env = simpy.Environment()
datacenter = Datacenter(env)

env.process(simulate(env, datacenter))
env.run(until=20)

class Broker:
    def __init__(self, env):
        self.env = env

    def request_service(self, vm, cloudlet):
        yield self.env.timeout(1)
        yield self.env.process(cloudlet.process_vm_request(vm))


class Cloudlet:
    def __init__(self, env):
        self.env = env
        self.server = simpy.Resource(env, capacity=1)

    def process_vm_request(self, vm):
        with self.server.request() as req:
            yield req
            print(f"Cloudlet: Virtual machine starts running at {self.env.now}")
            yield self.env.timeout(vm.runtime)
            print(f"Cloudlet: Virtual machine finishes at {self.env.now}")


class VirtualMachine:
    def __init__(self, env, runtime):
        self.env = env
        self.runtime = runtime


def simulate(env, broker, cloudlet):
    vm = VirtualMachine(env, runtime=10)
    yield env.process(broker.request_service(vm, cloudlet))


env = simpy.Environment()
broker = Broker(env)
cloudlet = Cloudlet(env)

env.process(simulate(env, broker, cloudlet))
env.run(until=20)

class Datacenter:
    def __init__(self, env, num_hosts):
        self.env = env
        self.hosts = [simpy.Resource(env, capacity=1) for _ in range(num_hosts)]

    def process_vm_request(self, vm):
        for host in self.hosts:
            with host.request() as req:
                yield req
                print(f"VM {vm.id} starts running at {self.env.now}")
                yield self.env.timeout(vm.runtime)
                print(f"VM {vm.id} finishes at {self.env.now}")

class VirtualMachine:
    id_counter = 0
    def __init__(self, env, runtime):
        self.id = VirtualMachine.id_counter
        VirtualMachine.id_counter += 1
        self.env = env
        self.runtime = runtime

def generate_workload(env, datacenter):
    while True:
        vm = VirtualMachine(env, runtime=random.randint(5, 15))
        yield env.process(datacenter.process_vm_request(vm))
        yield env.timeout(random.expovariate(1/10))

env = simpy.Environment()
datacenter = Datacenter(env, num_hosts=50)

env.process(generate_workload(env, datacenter))
env.run(until=50)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import simpy

# Load the dataset
file_path = '/dataset/dataset.txt'
data = pd.read_csv(file_path, header=None)
cpu_usage = data.values.flatten()

# Define parameters
sequence_length = 10


# Prepare the data for Linear Regression (previous n values predict next value)
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

# Define and train the Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predict
lr_predictions = model_lr.predict(X_test)

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
    prediction = model_lr.predict([last_n_values])
    predicted_range = (prediction - (0.2 * prediction), prediction + (0.2 * prediction))  # 20% range
    return prediction[0] * 100, predicted_range  # Multiply by 100 to scale the result


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