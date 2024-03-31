import numpy as np


class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size)
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # Initialize previous hidden state and cell state
        self.prev_hidden_state = np.zeros((hidden_size, 1))
        self.prev_cell_state = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        self.x = x

        self.concat = np.row_stack((self.prev_hidden_state, x))
        self.ft = self.sigmoid(np.dot(self.Wf, self.concat) + self.bf)
        self.it = self.sigmoid(np.dot(self.Wi, self.concat) + self.bi)
        self.ot = self.sigmoid(np.dot(self.Wo, self.concat) + self.bo)
        self.cct = self.tanh(np.dot(self.Wc, self.concat) + self.bc)

        self.cell_state = self.ft * self.prev_cell_state + self.it * self.cct
        self.hidden_state = self.ot * self.tanh(self.cell_state)

        self.output = np.dot(self.Wy, self.hidden_state) + self.by

        return self.output, self.hidden_state, self.cell_state

    def backward(self, d_output, d_hidden_state, d_cell_state, learning_rate):
        dWy = np.dot(d_output, self.hidden_state.T)
        dby = d_output
        d_hidden_state += np.dot(self.Wy.T, d_output)

        d_ot = d_hidden_state * self.tanh(self.cell_state)
        d_cell_state += d_hidden_state * self.ot * (1 - self.tanh(self.cell_state) ** 2)

        d_ft = d_cell_state * self.prev_cell_state
        d_prev_cell_state = d_cell_state * self.ft
        d_it = d_cell_state * self.cct
        d_cct = d_cell_state * self.it

        d_cct_raw = d_cct * (1 - self.cct**2)
        d_ft_raw = d_ft * self.ft * (1 - self.ft)
        d_it_raw = d_it * self.it * (1 - self.it)
        d_ot_raw = d_ot * self.ot * (1 - self.ot)

        d_Wc = np.dot(d_cct_raw, self.concat.T)
        d_bc = d_cct_raw
        d_Wf = np.dot(d_ft_raw, self.concat.T)
        d_bf = d_ft_raw
        d_Wi = np.dot(d_it_raw, self.concat.T)
        d_bi = d_it_raw
        d_Wo = np.dot(d_ot_raw, self.concat.T)
        d_bo = d_ot_raw

        d_concat = (
            np.dot(self.Wc.T, d_cct_raw)
            + np.dot(self.Wf.T, d_ft_raw)
            + np.dot(self.Wi.T, d_it_raw)
            + np.dot(self.Wo.T, d_ot_raw)
        )
        d_prev_hidden_state = d_concat[: self.hidden_size, :]
        d_x = d_concat[self.hidden_size :, :]

        # Update weights
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby
        self.Wc -= learning_rate * d_Wc
        self.bc -= learning_rate * d_bc
        self.Wf -= learning_rate * d_Wf
        self.bf -= learning_rate * d_bf
        self.Wi -= learning_rate * d_Wi
        self.bi -= learning_rate * d_bi
        self.Wo -= learning_rate * d_Wo
        self.bo -= learning_rate * d_bo

        # Update previous hidden state and cell state
        self.prev_hidden_state = self.hidden_state
        self.prev_cell_state = self.cell_state

        return d_prev_hidden_state, d_x
