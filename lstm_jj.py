import numpy as np
import jax.numpy as jnp
import jax

class LSTM:
    #define lstm cell using jnp, enabling gradient descent
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

        # Initialize hidden and cell states for the sequence
        hidden_state_sequence = []
        cell_state_sequence = []

        for i in range(x.shape[1]):  # Iterate over the sequence length
            # Reshape and concatenate previous hidden state
            reshaped_prev_hidden_state = self.prev_hidden_state.reshape(-1, 1)
            self.concat = np.column_stack((reshaped_prev_hidden_state, x[:, i:i+1]))

            # Compute gates
            self.ft = self.sigmoid(np.dot(self.Wf, self.concat) + self.bf)
            self.it = self.sigmoid(np.dot(self.Wi, self.concat) + self.bi)
            self.ot = self.sigmoid(np.dot(self.Wo, self.concat) + self.bo)
            self.cct = self.tanh(np.dot(self.Wc, self.concat) + self.bc)

            # Update cell state and hidden state
            self.cell_state = self.ft * self.prev_cell_state + self.it * self.cct
            self.hidden_state = self.ot * self.tanh(self.cell_state)

            # Save hidden and cell states for the sequence
            hidden_state_sequence.append(self.hidden_state)
            cell_state_sequence.append(self.cell_state)

            # Update previous hidden state and cell state for next iteration
            self.prev_hidden_state = self.hidden_state
            self.prev_cell_state = self.cell_state

        # Return the output of the last step in the sequence
        output = np.dot(self.Wy, self.hidden_state) + self.by
        return output, hidden_state_sequence[-1], cell_state_sequence[-1]
    
    def backward(self, dh_next, dc_next, f, i, c_hat, c, o, tanh_c):
        # Initialize gradients for each parameter
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWc = np.zeros_like(self.Wc)
        dWy = np.zeros_like(self.Wy)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbc = np.zeros_like(self.bc)
        dby = np.zeros_like(self.by)
        dh_prev = np.zeros_like(dh_next)
        dc_prev = np.zeros_like(dc_next)

        for t in reversed(range(self.x.shape[1])):
            hidden_state_sequence = []  # Define hidden_state_sequence variable
            for t in reversed(range(self.x.shape[1])):
                reshaped_prev_hidden_state = self.prev_hidden_state.reshape(-1, 1)
                concat = np.column_stack((reshaped_prev_hidden_state, self.x[:, t:t+1]))

                # Compute the gradient of the output layer w.r.t. the hidden state
                dWy += np.dot(dh_next, hidden_state_sequence[t].T)
                dby += dh_next

                # Compute the gradient of the hidden state w.r.t. the output gate
                dh = np.dot(self.Wy.T, dh_next) + dh_prev
                do = dh * tanh_c
                dot = do * o * (1 - o)

                # Compute the gradient of the cell state and candidate values
                dc = dc_next + dh * o * (1 - tanh_c ** 2)
                dcct = dc * i
            dcct = dcct * (1 - c_hat ** 2)

            # Compute the gradient of the input gate
            di = dc * c_hat
            dii = di * i * (1 - i)

            # Define c_prev
            c_prev = self.prev_cell_state

            # Compute the gradient of the forget gate
            df = dc * c_prev
            dff = df * f * (1 - f)

            # Compute the gradient of the cell state
            dc_prev = dc * f

            # Compute the gradients of the gates
            dWf += np.dot(dff, concat.T)
            dbf += dff
            dWi += np.dot(dii, concat.T)
            dbi += dii
            dWo += np.dot(dot, concat.T)
            dbo += dot
            dWc += np.dot(dcct, concat.T)
            dbc += dcct

            # Compute the gradient of the input
            dx = np.dot(df, self.Wf[:, :self.hidden_size].T) + np.dot(di, self.Wi[:, :self.hidden_size].T) + np.dot(do, self.Wo[:, :self.hidden_size].T) + np.dot(dcct, self.Wc[:, :self.hidden_size].T)

            # Compute the gradient of the previous hidden state
            dh_prev = np.dot(df, self.Wf[:, self.hidden_size:].T) + np.dot(di, self.Wi[:, self.hidden_size:].T) + np.dot(do, self.Wo[:, self.hidden_size:].T) + np.dot(dcct, self.Wc[:, self.hidden_size:].T)

        return dx, dh_prev, dc_prev, dWf, dbf, dWi, dbi, dWo, dbo, dWc, dbc, dWy, dby