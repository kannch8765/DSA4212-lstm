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
        self.prev_hidden_state = np.random.randn(hidden_size, 1)
        self.prev_cell_state = np.random.randn(hidden_size, 1)

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
            reshaped_prev_hidden_state = self.prev_hidden_state.reshape(-1, 1) # 1, 128
            self.concat = np.concatenate((reshaped_prev_hidden_state, x[:, i:i+1]))

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
        return output, hidden_state_sequence[-1], cell_state_sequence[-1], hidden_state_sequence, cell_state_sequence
    
    def backward(self, d_output, d_hidden_state, d_cell_state, hidden_state_sequence, cell_state_sequence, learning_rate=0.01):
        # Initialize gradients for each weight and bias
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

        # Initialize gradient for hidden state and cell state
        dh_next = np.zeros_like(d_hidden_state)
        dc_next = np.zeros_like(d_cell_state)

        for i in reversed(range(self.x.shape[1])):
            # Compute gradients at each step
            
            #  Compute gradient of output layer w.r.t. hidden state
            dWy += np.dot(d_output, hidden_state_sequence[i].T)
            dby += d_output

            # Compute gradient of hidden state w.r.t. output gate
            dh = np.dot(self.Wy.T, d_output) + dh_next
            do = dh * self.tanh(cell_state_sequence[i])
            dot = do * self.ot * (1 - self.ot)

            # Compute gradient of hidden state w.r.t. cell state
            dc = d_cell_state + dh * self.ot * (1 - self.tanh(cell_state_sequence[i]) ** 2)

            # Compute gradient of gates
            dft = dc * self.prev_cell_state * self.ft * (1 - self.ft)
            dWf += np.dot(dft, self.concat.T)
            dbf += dft
            dprev_cell_state = dc * self.ft

            dit = dc * self.cct * self.it * (1 - self.it)
            dWi += np.dot(dit, self.concat.T)
            dbi += dit
            dcct = dc * self.it
            dcctt = dcct * (1 - self.cct ** 2)
            dWc += np.dot(dcctt, self.concat.T)
            dbc += dcctt

            # Compute gradient of forget gate
            df = dc * self.prev_cell_state * self.ft * (1 - self.ft)

            # Compute gradient of input gate
            di = dc * self.cct * self.it * (1 - self.it)

            # Compute gradient of previous hidden state
            dprev_hidden_state = np.dot(df, self.Wf[:, :self.hidden_size].T) \
                            + np.dot(di, self.Wi[:, :self.hidden_size].T) \
                            + np.dot(do, self.Wo[:, :self.hidden_size].T) \
                            + np.dot(dcct, self.Wc[:, :self.hidden_size].T)
            
            # Compute gradient of previous cell state
            dprev_cell_state += dc * self.ft

            # Update weights and biases using gradients & learning rate
            self.Wf -= learning_rate * dWf
            self.Wi -= learning_rate * dWi
            self.Wo -= learning_rate * dWo
            self.Wc -= learning_rate * dWc
            self.Wy -= learning_rate * dWy

            self.bf -= learning_rate * dbf
            self.bi -= learning_rate * dbi
            self.bo -= learning_rate * dbo
            self.bc -= learning_rate * dbc
            self.by -= learning_rate * dby

            # Update hidden state and cell state
            self.prev_hidden_state = hidden_state_sequence[i]
            self.prev_cell_state = cell_state_sequence[i]
