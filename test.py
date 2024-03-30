import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
data = pd.read_csv('daily_climate_data.csv')

# Preprocessing
# Selecting relevant features
data = data[['date', 'meantemp']]

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

# Sort by date
data.sort_values(by='date', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data['meantemp_scaled'] = scaler.fit_transform(data['meantemp'].values.reshape(-1, 1))

# Create sequences of data for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Define sequence length
sequence_length = 7

# Create sequences
X, y = create_sequences(data['meantemp_scaled'].values, sequence_length)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential([
    LSTM(128, input_shape=(sequence_length, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print("Mean Squared Error (MSE):", mse)

# Predictions
predictions = model.predict(X_test)

# Plot actual vs predicted
import matplotlib.pyplot as plt
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Mean Temperature (Scaled)')
plt.legend()
plt.show()
