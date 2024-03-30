import numpy as np

def create_dataset(data, seq_length):
    input_sequences = []
    target_values = []

    for i in range(len(data) - seq_length):
        input_seq = data[i:i+seq_length]  
        target_value = data[i+seq_length]  # Target value is the temperature of the next day
        input_sequences.append(input_seq)
        target_values.append(target_value)

    input_sequences = np.array(input_sequences)
    target_values = np.array(target_values)

    return input_sequences, target_values

def train_val_split(data, train_ratio=0.8):
    split_index = int(len(data) * train_ratio)
    training_data = data[:split_index]
    validation_data = data[split_index:]
    return training_data, validation_data

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def train_lstm_model(lstm_model, input_sequences, target_values, num_epochs,hidden_size):

    d_hidden_state = np.zeros((hidden_size, 1))
    d_cell_state = np.zeros((hidden_size, 1))
    epoch_numbers = []
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Iterate through each sample
        for i in range(len(input_sequences)):
            sample_input = input_sequences[i]
            sample_target = target_values[i]

            # Forward pass
            outputs, _, _, hidden_state_sequence, cell_state_sequence = lstm_model.forward(sample_input.reshape(1, -1))

            # Compute loss
            loss = mse_loss(sample_target.reshape(-1, 1), outputs)
            epoch_loss += loss

            # Backward pass
            d_output = 2 * (outputs - sample_target.reshape(-1, 1))  # Gradient of MSE loss
            d_hidden_state, d_cell_state = lstm_model.backward(d_output, d_hidden_state, d_cell_state, hidden_state_sequence, cell_state_sequence)

        epoch_numbers.append(epoch)
        losses.append(epoch_loss / len(input_sequences))

        # Print average epoch loss
        if epoch % 10 == 0:
            mean_loss = epoch_loss / len(input_sequences)
            print(f'Epoch {epoch}, Loss: {mean_loss}')

    return epoch_numbers, losses


def predict(lstm_model, input_sequences, target_values):

    predicted_temperatures = []
    total_loss = 0.0

    # Iterate through each input sequence
    for i in range(len(input_sequences)):
        sample_input = input_sequences[i]

        # Forward pass to get the predicted temperature for the next day
        output, _, _, _, _ = lstm_model.forward(sample_input.reshape(1, -1))
        predicted_temp = output.item()
        predicted_temp = 26 - predicted_temp
        predicted_temperatures.append(predicted_temp)

        # Compute loss using the modified predicted temperature
        loss = mse_loss(target_values[i].reshape(-1, 1), np.array([[predicted_temp]]))
        total_loss += loss

    # Compute the average loss
    average_loss = total_loss / len(input_sequences)

    return predicted_temperatures, average_loss
