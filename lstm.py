import numpy as np

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
input_size = 3
hidden_size = 4

lstm_cell = LSTMCell(input_size, hidden_size)
x = np.random.randn(input_size, 1)
prev_hidden_state = np.random.randn(hidden_size, 1)
prev_cell_state = np.random.randn(hidden_size, 1)

hidden_state, cell_state = lstm_cell.forward(x, prev_hidden_state, prev_cell_state)
print("Hidden State:", hidden_state)
print("Cell State:", cell_state)
