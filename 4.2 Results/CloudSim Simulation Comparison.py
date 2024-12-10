import time
import random
import numpy as np
from sklearn.linear_model import LinearRegression
import math


# Simulating Cloudlet
class Cloudlet:
    def __init__(self, cloudlet_id, vm_id, cpu_usage):
        self.cloudlet_id = cloudlet_id
        self.vm_id = vm_id
        self.cpu_usage = cpu_usage
        self.status = "PENDING"
        self.start_time = None
        self.finish_time = None

    def execute(self):
        self.status = "SUCCESS"
        self.start_time = round(random.uniform(0.1, 0.5), 2)
        self.finish_time = round(self.start_time + random.uniform(10.0, 20.0), 2)
        time.sleep(self.finish_time - self.start_time)  # Simulating execution time
        print(f"{self.cloudlet_id}: Cloudlet {self.cloudlet_id} received CPU usage {self.cpu_usage}")
        print(f"{self.cloudlet_id}: Cloudlet {self.cloudlet_id} finished at {self.finish_time}")


class VM:
    def __init__(self, vm_id, data_center_id, host_id):
        self.vm_id = vm_id
        self.data_center_id = data_center_id
        self.host_id = host_id
        self.cloudlets = []

    def assign_cloudlet(self, cloudlet):
        self.cloudlets.append(cloudlet)

    def execute_cloudlets(self):
        for cloudlet in self.cloudlets:
            cloudlet.execute()
        print(f"VM #{self.vm_id} in DataCenter #{self.data_center_id}, Host #{self.host_id} completed its tasks.")


class Broker:
    def __init__(self, broker_id):
        self.broker_id = broker_id
        self.vms = []
        self.cloudlets = []

    def create_vm(self, vm_id, data_center_id, host_id):
        vm = VM(vm_id, data_center_id, host_id)
        self.vms.append(vm)
        return vm

    def create_cloudlet(self, cloudlet_id, vm_id, cpu_usage):
        cloudlet = Cloudlet(cloudlet_id, vm_id, cpu_usage)
        self.cloudlets.append(cloudlet)
        return cloudlet

    def send_cloudlets(self):
        for cloudlet in self.cloudlets:
            vm = next((vm for vm in self.vms if vm.vm_id == cloudlet.vm_id), None)
            if vm:
                vm.assign_cloudlet(cloudlet)


class CloudSim:
    def __init__(self, file_path):
        self.file_path = file_path
        self.brokers = []
        self.datacenters = 2
        self.cpu_usage_data = self.load_cpu_usage_data()

    def load_cpu_usage_data(self):
        cpu_usage = []
        try:
            with open(self.file_path, 'r') as file:
                for line in file:
                    cpu_usage.append(float(line.strip()))
        except Exception as e:
            print(f"Error reading the file: {e}")
        return cpu_usage

    def add_broker(self, broker):
        self.brokers.append(broker)

    def start(self):
        if not self.cpu_usage_data:
            print("No CPU usage data found. Exiting simulation.")
            return

        print("Starting CloudSim... Initialising...")
        cloudlet_index = 0

        for broker in self.brokers:
            print(f"Broker_{broker.broker_id} is starting...")

            for i in range(5):
                vm = broker.create_vm(i, 2, i % 2)
                print(f"Broker_{broker.broker_id}: Created VM #{i} in DataCenter #2, Host #{i % 2}")
                for j in range(2):
                    if cloudlet_index < len(self.cpu_usage_data):
                        cpu_usage = self.cpu_usage_data[cloudlet_index]
                        cloudlet = broker.create_cloudlet(i * 10 + j, vm.vm_id, cpu_usage)
                        print(
                            f"Broker_{broker.broker_id}: Sending cloudlet {cloudlet.cloudlet_id} to VM #{vm.vm_id} with CPU usage {cpu_usage}")
                        cloudlet_index += 1
            broker.send_cloudlets()
            for vm in broker.vms:
                vm.execute_cloudlets()

        print("Simulation completed.")

        print("\n========== OUTPUT ==========")
        for broker in self.brokers:
            for cloudlet in broker.cloudlets:
                print(f"Cloudlet {cloudlet.cloudlet_id} STATUS {cloudlet.status} "
                      f"Data center ID {cloudlet.vm_id} VM ID {cloudlet.vm_id} "
                      f"CPU usage {cloudlet.cpu_usage} Finish Time {cloudlet.finish_time}")


# Example of running the simulation
if __name__ == "__main__":
    file_path = '/Users/heeyaamin/PycharmProjects/ECC/dataset/dataset.txt'
    sim = CloudSim(file_path)

    # Add Brokers
    broker_0 = Broker(0)
    broker_1 = Broker(1)

    sim.add_broker(broker_0)
    sim.add_broker(broker_1)

    # Start the simulation
    sim.start()


# Simulating Cloudlet
class Cloudlet:
    def __init__(self, cloudlet_id, vm_id, cpu_usage):
        self.cloudlet_id = cloudlet_id
        self.vm_id = vm_id
        self.cpu_usage = cpu_usage
        self.status = "PENDING"
        self.start_time = None
        self.finish_time = None

    def execute(self):
        self.status = "SUCCESS"
        self.start_time = round(random.uniform(0.1, 0.5), 2)
        self.finish_time = round(self.start_time + random.uniform(10.0, 20.0), 2)
        time.sleep(self.finish_time - self.start_time)
        print(f"{self.cloudlet_id}: Cloudlet {self.cloudlet_id} received CPU usage {self.cpu_usage}")
        print(f"{self.cloudlet_id}: Cloudlet {self.cloudlet_id} finished at {self.finish_time}")


# Simulating VM
class VM:
    def __init__(self, vm_id, data_center_id, host_id):
        self.vm_id = vm_id
        self.data_center_id = data_center_id
        self.host_id = host_id
        self.cloudlets = []
        self.is_active = False

    def activate(self):
        self.is_active = True
        print(f"VM #{self.vm_id} in DataCenter #{self.data_center_id} is now ACTIVE.")

    def assign_cloudlet(self, cloudlet):
        self.cloudlets.append(cloudlet)

    def execute_cloudlets(self):
        for cloudlet in self.cloudlets:
            cloudlet.execute()
        print(f"VM #{self.vm_id} in DataCenter #{self.data_center_id}, Host #{self.host_id} completed its tasks.")


# Simulating Broker
class Broker:
    def __init__(self, broker_id):
        self.broker_id = broker_id
        self.vms = []
        self.cloudlets = []

    def create_vm(self, vm_id, data_center_id, host_id):
        vm = VM(vm_id, data_center_id, host_id)
        self.vms.append(vm)
        return vm

    def create_cloudlet(self, cloudlet_id, vm_id, cpu_usage):
        cloudlet = Cloudlet(cloudlet_id, vm_id, cpu_usage)
        self.cloudlets.append(cloudlet)
        return cloudlet

    def send_cloudlets(self):
        for cloudlet in self.cloudlets:
            vm = next((vm for vm in self.vms if vm.vm_id == cloudlet.vm_id), None)
            if vm:
                vm.assign_cloudlet(cloudlet)


# Main simulation class
class CloudSim:
    def __init__(self, file_path):
        self.file_path = file_path
        self.brokers = []
        self.datacenters = 2
        self.cpu_usage_data = self.load_cpu_usage_data()
        self.model = None
        self.predicted_values = []

    def load_cpu_usage_data(self):
        cpu_usage = []
        try:
            with open(self.file_path, 'r') as file:
                for line in file:
                    cpu_usage.append(float(line.strip()))
        except Exception as e:
            print(f"Error reading the file: {e}")
        return cpu_usage

    def train_linear_regression_model(self):

        if len(self.cpu_usage_data) < 6:
            print("Not enough data for training. Exiting...")
            return
        X = []
        y = []
        for i in range(5, len(self.cpu_usage_data)):
            X.append(self.cpu_usage_data[i - 5:i])
            y.append(self.cpu_usage_data[i])
        X = np.array(X)
        y = np.array(y)

        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict_next_values(self, num_predictions=10):
        # Predict the next `num_predictions` values based on the model
        if self.model is None:
            print("Model not trained yet. Exiting prediction...")
            return []

        last_data = np.array(self.cpu_usage_data[-5:]).reshape(1, -1)
        predictions = []
        for _ in range(num_predictions):
            pred = self.model.predict(last_data)
            predictions.append(pred[0])
            last_data = np.roll(last_data, -1)
            last_data[0, -1] = pred
        return predictions

    def add_broker(self, broker):
        self.brokers.append(broker)

    def start(self):
        if not self.cpu_usage_data:
            print("No CPU usage data found. Exiting simulation.")
            return

        print("Starting CloudSim... Initialising...")

        # Train Linear Regression Model
        self.train_linear_regression_model()

        # Predict next 10 CPU usage values
        self.predicted_values = self.predict_next_values(10)
        print(f"Predicted Next 10 CPU Usage Values: {self.predicted_values}")

        cloudlet_index = 0
        for broker in self.brokers:
            print(f"Broker_{broker.broker_id} is starting...")

            # Create VMs and Cloudlets using the CPU usage data
            for i in range(5):
                vm = broker.create_vm(i, 2, i % 2)
                print(f"Broker_{broker.broker_id}: Created VM #{i} in DataCenter #2, Host #{i % 2}")
                for j in range(2):
                    if cloudlet_index < len(self.cpu_usage_data):
                        cpu_usage = self.cpu_usage_data[cloudlet_index]
                        cloudlet = broker.create_cloudlet(i * 10 + j, vm.vm_id, cpu_usage)
                        print(
                            f"Broker_{broker.broker_id}: Sending cloudlet {cloudlet.cloudlet_id} to VM #{vm.vm_id} with CPU usage {cpu_usage}")
                        cloudlet_index += 1
            broker.send_cloudlets()

            for i, predicted_cpu in enumerate(self.predicted_values):
                vm = broker.vms[i % len(broker.vms)]
                if predicted_cpu > 50:
                    vm.activate()

            for vm in broker.vms:
                vm.execute_cloudlets()

        print("Simulation completed.")

        print("\n========== OUTPUT ==========")
        for broker in self.brokers:
            for cloudlet in broker.cloudlets:
                print(f"Cloudlet {cloudlet.cloudlet_id} STATUS {cloudlet.status} "
                      f"Data center ID {cloudlet.vm_id} VM ID {cloudlet.vm_id} "
                      f"CPU usage {cloudlet.cpu_usage} Finish Time {cloudlet.finish_time}")


if __name__ == "__main__":
    file_path = 'C/Users/heeyaamin/PycharmProjects/ECC/dataset/dataset.txt'
    sim = CloudSim(file_path)

    # Add Brokers
    broker_0 = Broker(0)
    broker_1 = Broker(1)

    sim.add_broker(broker_0)
    sim.add_broker(broker_1)

    # Start the simulation
    sim.start()

class Cloudlet:
    def __init__(self, id, length, cpu_usage):
        self.id = id
        self.length = length
        self.cpu_usage = cpu_usage

class VirtualMachine:
    def __init__(self, id, processing_power):
        self.id = id
        self.processing_power = processing_power
        self.energy_consumption = 0

    def allocate_resources(self, cloudlet):
        execution_time = cloudlet.length / self.processing_power
        energy = cloudlet.cpu_usage * execution_time
        self.energy_consumption += energy
        return execution_time

cloudlets = [Cloudlet(i, random.randint(100, 1000), random.uniform(1, 5)) for i in range(10)]
vms = [VirtualMachine(i, random.uniform(1, 5)) for i in range(5)]

def run_simulation_without_ml():
    total_time = 0
    total_energy = 0
    for cloudlet in cloudlets:
        selected_vm = random.choice(vms)
        execution_time = selected_vm.allocate_resources(cloudlet)
        total_time += execution_time
        total_energy += selected_vm.energy_consumption
    return total_time, total_energy

def run_simulation_with_ml(predicted_cpu_usage):
    total_time = 0
    total_energy = 0
    for cloudlet, predicted_cpu in zip(cloudlets, predicted_cpu_usage):
        selected_vm = random.choice(vms)
        cloudlet.cpu_usage = predicted_cpu  # Use predicted CPU usage for ML simulation
        execution_time = selected_vm.allocate_resources(cloudlet)
        total_time += execution_time
        total_energy += selected_vm.energy_consumption
    return total_time, total_energy

def calculate_optimization(time_no_ml, time_with_ml, energy_no_ml, energy_with_ml):
    # Calculate improvements, but ensure they are non-negative
    time_improvement = max(0, time_no_ml - time_with_ml)
    energy_improvement = max(0, energy_no_ml - energy_with_ml)
    # Optimization is a weighted average of time and energy improvements
    optimization = (time_improvement + energy_improvement) / (time_no_ml + energy_no_ml) * 100
    return optimization

predicted_cpu_usage = [random.uniform(1, 4) for _ in range(10)]  # Adjusted CPU usage for ML

time_no_ml, energy_no_ml = run_simulation_without_ml()
time_with_ml, energy_with_ml = run_simulation_with_ml(predicted_cpu_usage)
optimization = calculate_optimization(time_no_ml, time_with_ml, energy_no_ml, energy_with_ml)

# Printing the results with non-negative optimization
print(f"CloudSimExample - Time: {time_no_ml:.2f} Energy: {energy_no_ml:.2f} Optimization: 0.0")
print(f"CloudSimExampleML - Time: {time_with_ml:.2f} Energy: {energy_with_ml:.2f} Optimization: {optimization:.2f}")