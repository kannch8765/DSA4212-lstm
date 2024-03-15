import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, num_layers):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize LSTM layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(LSTMCell(input_size, hidden_size))
            input_size = hidden_size  # Update input size for next layer
        
    def forward(self, x, prev_hidden_states, prev_cell_states):
        hidden_states = []
        cell_states = []
        
        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                prev_hidden_state = prev_hidden_states[layer_idx]
                prev_cell_state = prev_cell_states[layer_idx]
            else:
                prev_hidden_state = hidden_states[-1]
                prev_cell_state = cell_states[-1]
            
            # Forward pass through current layer
            hidden_state, cell_state = self.layers[layer_idx].forward(x, prev_hidden_state, prev_cell_state)
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)
            
            # Output of current layer becomes input to the next layer
            x = hidden_state
        
        return hidden_states, cell_states

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_f = np.zeros((hidden_size, 1))
        
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_i = np.zeros((hidden_size, 1))
        
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_c = np.zeros((hidden_size, 1))
        
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_o = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, prev_hidden_state, prev_cell_state):
        # Concatenate input and previous hidden state
        concat_input = np.vstack((prev_hidden_state, x))
        
        # Forget gate
        f = self.sigmoid(np.dot(self.W_f, concat_input) + self.b_f)
        
        # Input gate
        i = self.sigmoid(np.dot(self.W_i, concat_input) + self.b_i)
        
        # Cell state (candidate)
        c_tilde = self.tanh(np.dot(self.W_c, concat_input) + self.b_c)
        
        # Update cell state
        cell_state = f * prev_cell_state + i * c_tilde
        
        # Output gate
        o = self.sigmoid(np.dot(self.W_o, concat_input) + self.b_o)
        
        # Hidden state
        hidden_state = o * self.tanh(cell_state)
        
        return hidden_state, cell_state

# Example usage:
input_size = 1
hidden_size = 4
num_layers = 2

lstm_model = LSTM(input_size, hidden_size, num_layers)

# Now you can use lstm_model.forward() for forward pass through the LSTM layers.
