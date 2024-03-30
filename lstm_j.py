import jax.numpy as jnp
import jax
from jax import grad

class LSTM_JAX:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        key = jax.random.PRNGKey(0)  # Initialize a random key
        self.Wf = jax.random.normal(key, (hidden_size, input_size + hidden_size))
        self.Wi = jax.random.normal(key, (hidden_size, input_size + hidden_size))
        self.Wo = jax.random.normal(key, (hidden_size, input_size + hidden_size))
        self.Wc = jax.random.normal(key, (hidden_size, input_size + hidden_size))
        self.Wy = jax.random.normal(key, (output_size, hidden_size))
        self.bf = jnp.zeros((hidden_size, 1))
        self.bi = jnp.zeros((hidden_size, 1))
        self.bo = jnp.zeros((hidden_size, 1))
        self.bc = jnp.zeros((hidden_size, 1))
        self.by = jnp.zeros((output_size, 1))

        key, subkey = jax.random.split(key)
        self.prev_hidden_state = jax.nn.initializers.orthogonal()(subkey, (hidden_size, 1))
        self.prev_cell_state = jax.nn.initializers.orthogonal()(subkey, (hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))

    def tanh(self, x):
        return jnp.tanh(x)

    def forward(self, x, h_prev, c_prev):
        # Initialize hidden and cell states for the sequence
        hidden_state_sequence = []
        cell_state_sequence = []

        prev_hidden_state = h_prev
        prev_cell_state = c_prev

        for i in range(x.shape[0]):  # Iterate over the sequence length
            # Reshape and concatenate previous hidden state
            reshaped_prev_hidden_state = prev_hidden_state.reshape(-1, 1)
            concat = jnp.concatenate((reshaped_prev_hidden_state, x[i:i+1]), axis=0)

            # Compute gates
            ft = self.sigmoid(jnp.dot(self.Wf, concat) + self.bf)
            it = self.sigmoid(jnp.dot(self.Wi, concat) + self.bi)
            ot = self.sigmoid(jnp.dot(self.Wo, concat) + self.bo)
            cct = self.tanh(jnp.dot(self.Wc, concat) + self.bc)

            # Update cell state and hidden state
            cell_state = ft * prev_cell_state + it * cct
            hidden_state = ot * self.tanh(cell_state)

            # Save hidden and cell states for the sequence
            hidden_state_sequence.append(hidden_state)
            cell_state_sequence.append(cell_state)

            # Update previous hidden state and cell state for next iteration
            prev_hidden_state = hidden_state
            prev_cell_state = cell_state

        # Return the output of the last step in the sequence
        output = jnp.dot(self.Wy, hidden_state) + self.by
        return output, hidden_state_sequence, cell_state_sequence

    
    def backward(self, x, y, hidden_state_sequence, cell_state_sequence, learning_rate, clip_value):
        # Initialize gradients
        dWf = jnp.zeros_like(self.Wf)
        dWi = jnp.zeros_like(self.Wi)
        dWo = jnp.zeros_like(self.Wo)
        dWc = jnp.zeros_like(self.Wc)
        dWy = jnp.zeros_like(self.Wy)
        dbf = jnp.zeros_like(self.bf)
        dbi = jnp.zeros_like(self.bi)
        dbo = jnp.zeros_like(self.bo)
        dbc = jnp.zeros_like(self.bc)
        dby = jnp.zeros_like(self.by)
        dprev_hidden_state = jnp.zeros_like(self.prev_hidden_state)
        dprev_cell_state = jnp.zeros_like(self.prev_cell_state)

        loss = 0  # Initialize loss
        
        # Iterate over the sequence
        for i in reversed(range(len(hidden_state_sequence))):
            # Compute loss for each step and accumulate
            loss += self.loss(x[i:i+1], y[:, i:i+1])

            # Compute gradients for the current step
            d_output = 2 * (self.forward(x[i:i+1], self.prev_hidden_state, self.prev_cell_state)[0] - y[:, i:i+1])
            d_output = d_output.reshape(self.hidden_size, 1)
            dh_next = d_output @ self.Wy.T + dprev_hidden_state
            dc_next = dprev_cell_state

            reshaped_prev_hidden_state = self.prev_hidden_state.reshape(-1, 1)
            concat = jnp.column_stack((reshaped_prev_hidden_state, x[:, i:i+1]))

            # Compute gate derivatives
            ft = self.sigmoid(jnp.dot(self.Wf, concat) + self.bf)
            it = self.sigmoid(jnp.dot(self.Wi, concat) + self.bi)
            ot = self.sigmoid(jnp.dot(self.Wo, concat) + self.bo)
            cct = self.tanh(jnp.dot(self.Wc, concat) + self.bc)

            # Compute the derivative of the hidden state with respect to the output
            dh = jnp.dot(self.Wy.T, dh_next) + dprev_hidden_state
            dc = dc_next + dh_next * ot * (1 - self.tanh(cell_state_sequence[i]) ** 2)

            # Compute the derivative of the cell state
            dcct = dc * it
            dit = dc * cct
            dft = dc * cell_state_sequence[i - 1]
            dot = dh * self.tanh(cell_state_sequence[i])

            # Compute the derivative of the activations
            dit = dit * it * (1 - it)
            dft = dft * ft * (1 - ft)
            dot = dot * ot * (1 - ot)
            dcct = dcct * (1 - cct ** 2)

            # Accumulate gradients
            dWf += jnp.dot(dft, concat.T)
            dWi += jnp.dot(dit, concat.T)
            dWo += jnp.dot(dot, concat.T)
            dWc += jnp.dot(dcct, concat.T)
            dbf += dft
            dbi += dit
            dbo += dot
            dbc += dcct

            # Update previous hidden state and cell state for next iteration
            dprev_hidden_state = dh
            dprev_cell_state = dc

        # Clip gradients
        dWf = jnp.clip(dWf, -clip_value, clip_value)
        dWi = jnp.clip(dWi, -clip_value, clip_value)
        dWo = jnp.clip(dWo, -clip_value, clip_value)
        dWc = jnp.clip(dWc, -clip_value, clip_value)
        dWy = jnp.clip(dWy, -clip_value, clip_value)
        dbf = jnp.clip(dbf, -clip_value, clip_value)
        dbi = jnp.clip(dbi, -clip_value, clip_value)
        dbo = jnp.clip(dbo, -clip_value, clip_value)
        dbc = jnp.clip(dbc, -clip_value, clip_value)
        dby = jnp.clip(dby, -clip_value, clip_value)

        # Update parameters
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

        # Return gradients and loss
        grads = {'Wf': dWf, 'Wi': dWi, 'Wo': dWo, 'Wc': dWc, 'Wy': dWy,
                'bf': dbf, 'bi': dbi, 'bo': dbo, 'bc': dbc, 'by': dby}
        
        return grads, loss

    
    def loss(self, x, y):
    # Call forward method with initial hidden state and cell state as zeros
        output, _, _ = self.forward(x, jnp.zeros((self.hidden_size, 1)), jnp.zeros((self.hidden_size, 1)))
        loss = jnp.sum((output - y) ** 2)
        return loss
