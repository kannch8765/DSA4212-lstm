import numpy as np

class LSTM_T:
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

        return h, c, o
    
    def backward(self, x, h_prev, c_prev, dh_next, dc_next, f, i, c_hat, c, o, tanh_c):
        # Concatenate input and previous hidden state
        concat_input = np.concatenate((h_prev, x), axis=1)

        # Gradient of loss w.r.t. output gate weights and biases
        dW_o = np.dot(concat_input.T, dh_next * tanh_c * o * (1 - o))
        db_o = np.sum(dh_next * tanh_c * o * (1 - o), axis=0, keepdims=True)

        # Gradient of loss w.r.t. cell state
        d_c = dh_next * o * (1 - tanh_c ** 2) + dc_next

        # Gradient of loss w.r.t. input gate weights and biases
        dW_i = np.dot(concat_input.T, d_c * i * (1 - i))
        db_i = np.sum(d_c * i * (1 - i), axis=0, keepdims=True)

        # Gradient of loss w.r.t. candidate values weights and biases
        dW_c = np.dot(concat_input.T, d_c * f * (1 - c_hat ** 2))
        db_c = np.sum(d_c * f * (1 - c_hat ** 2), axis=0, keepdims=True)

        # Gradient of loss w.r.t. forget gate weights and biases
        dW_f = np.dot(concat_input.T, d_c * c_prev * (1 - f) * f)
        db_f = np.sum(d_c * c_prev * (1 - f) * f, axis=0, keepdims=True)

        # Gradient of loss w.r.t. previous hidden state
        d_h_prev = np.dot(d_c * f, self.W_f.T) + np.dot(d_c * i, self.W_i.T) + np.dot(d_c * o * (1 - tanh_c ** 2), self.W_o.T)

        # Gradient of loss w.r.t. previous cell state
        d_c_prev = d_c * f

        # Gradient of loss w.r.t. input
        d_x = np.dot(d_c * f, self.W_f[:, :self.hidden_size].T) + np.dot(d_c * i, self.W_i[:, :self.hidden_size].T) + np.dot(d_c * o * (1 - tanh_c ** 2), self.W_o[:, :self.hidden_size].T)

        return d_x, d_h_prev, d_c_prev, dW_f, db_f, dW_i, db_i, dW_c, db_c, dW_o, db_o
    
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

