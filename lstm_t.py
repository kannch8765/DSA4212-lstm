import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weights and biases
        self.W_f = np.random.randn(input_size + hidden_size, hidden_size)  # Forget gate
        self.b_f = np.zeros((1, hidden_size))

        self.W_i = np.random.randn(input_size + hidden_size, hidden_size)  # Input gate
        self.b_i = np.zeros((1, hidden_size))

        self.W_c = np.random.randn(input_size + hidden_size, hidden_size)  # Candidate values
        self.b_c = np.zeros((1, hidden_size))

        self.W_o = np.random.randn(input_size + hidden_size, hidden_size)  # Output gate
        self.b_o = np.zeros((1, hidden_size))

    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        concat_input = np.concatenate((h_prev, x), axis=1)

        # Forget gate
        f = sigmoid(np.dot(concat_input, self.W_f) + self.b_f)

        # Input gate
        i = sigmoid(np.dot(concat_input, self.W_i) + self.b_i)

        # Candidate values
        c_hat = tanh(np.dot(concat_input, self.W_c) + self.b_c)

        # Update cell state
        c = f * c_prev + i * c_hat

        # Output gate
        o = sigmoid(np.dot(concat_input, self.W_o) + self.b_o)

        # Update hidden state
        h = o * tanh(c)

        return h, c