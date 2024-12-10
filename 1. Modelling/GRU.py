import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r'/Users/heeyaamin/ECC_project/dataset/dataset.txt', header=None, names=['CPU_Usage'])

# Feature Engineering: Adding a moving average
data['Moving_Avg'] = data['CPU_Usage'].rolling(window=10).mean().fillna(0)
features = data[['CPU_Usage', 'Moving_Avg']].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)


# Prepare data for GRU
def create_dataset(dataset, time_step=10):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 10
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Split data into training and test sets
train_size = 80000
test_size = 20000
X_train, X_test = X[:train_size], X[train_size:train_size + test_size]
y_train, y_test = y[:train_size], y[train_size:train_size + test_size]


# Custom loss function to penalize high errors
def custom_loss(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true))
    penalty = K.mean(K.square(K.maximum(0., y_true - y_pred)))  # Penalize large errors
    return loss + 0.5 * penalty


# Build and train the GRU model
model = Sequential()
model.add(GRU(100, return_sequences=True, input_shape=(time_step, 2)))  # Increased units
model.add(GRU(100, return_sequences=False))  # Increased units
model.add(Dense(1))
model.compile(optimizer='adam', loss=custom_loss)

model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)  # Increased epochs

# Predict on the test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_test_inv = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1))]))[:, 0]
y_pred_inv = scaler.inverse_transform(np.hstack([y_pred, np.zeros((y_pred.shape[0], 1))]))[:, 0]

# Evaluate the model
mse = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_test_inv)), y_test_inv, label='Actual Data')
plt.plot(range(len(y_pred_inv)), y_pred_inv, label='Predicted Data', color='red')
plt.title('GRU Model: Actual vs Predicted Data')
plt.xlabel('Time Steps')
plt.ylabel('CPU Usage')
plt.legend()
plt.show()

# Predict next 10 values using the last sequence from the test set
last_sequence = X_test[-1].reshape(1, time_step, 2)
future_predictions = []

for _ in range(10):
    pred = model.predict(last_sequence)
    future_predictions.append(pred[0, 0])

    # Update the sequence
    last_sequence = np.roll(last_sequence, shift=-1, axis=1)
    last_sequence[0, -1, 0] = pred[0, 0]
    # Calculate new moving average manually for future predictions
    moving_avg = np.mean(last_sequence[0, :, 0])
    last_sequence[0, -1, 1] = moving_avg  # Update moving average

# Rescale future predictions manually
min_val, max_val = scaler.data_min_[0], scaler.data_max_[0]
future_predictions = np.clip(future_predictions, min_val, max_val)
future_predictions = future_predictions * (max_val - min_val) + min_val

# Print next 10 predicted values
print(f'Next 10 predicted values: {future_predictions}')