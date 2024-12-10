import numpy as np
from sklearn.linear_model import LinearRegression

def collect_data():
    vm_cpu_utilization = np.random.rand(10)
    vm_memory_utilization = np.random.rand(10)
    task_arrivals = np.random.randint(0, 100, size=(10,))
    return vm_cpu_utilization, vm_memory_utilization, task_arrivals
vm_cpu_utilization, vm_memory_utilization, task_arrivals = collect_data()
print(collect_data())


def train_regression_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model
X_train = np.column_stack((vm_cpu_utilization, vm_memory_utilization, task_arrivals))
y_train = np.random.rand(10)

regression_model = train_regression_model(X_train, y_train)

def allocate_resources(vm_cpu_utilization, vm_memory_utilization, task_arrivals, model):
    X = np.column_stack((vm_cpu_utilization, vm_memory_utilization, task_arrivals))
    predicted_response_time = model.predict(X)
    return predicted_response_time

predicted_response_time = allocate_resources(vm_cpu_utilization, vm_memory_utilization, task_arrivals, regression_model)

print(predicted_response_time)
