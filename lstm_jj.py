import numpy as np
import jax.numpy as jnp
import jax

class LSTM:
    #define lstm cell using jnp, enabling gradient descent
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

    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))
    
    def tanh(self, x):
        return jnp.tanh(x)
    
    def loss(self, y, y_pred):
        return jnp.mean(jnp.square(y - y_pred))
    
    def forward(self, x, h_prev, c_prev):     
        # Concatenate input and previous hidden state
        concat_input = jnp.concatenate((h_prev, x), axis=1)

        # Forget gate
        f = self.sigmoid(jnp.dot(concat_input, self.W_f) + self.b_f)

        # Input gate
        i = self.sigmoid(jnp.dot(concat_input, self.W_i) + self.b_i)

        # Candidate values
        c_hat = self.tanh(jnp.dot(concat_input, self.W_c) + self.b_c)

        # Update cell state
        c = f * c_prev + i * c_hat

        # Output gate
        o = self.sigmoid(jnp.dot(concat_input, self.W_o) + self.b_o)

        # Update hidden state
        h = o * self.tanh(c)

        return h, c, o, f, i, c_hat, c, o, self.tanh(c)
    
    def backward(self, x, h_prev, c_prev, dh_next, dc_next, f, i, c_hat, c, o, tanh_c, learning_rate=0.01):
        # Gradient of loss w.r.t. output gate weights and biases
        
        concat_input = jnp.concatenate((h_prev, x), axis=1)
        dW_o = jnp.dot(concat_input.T, dh_next * tanh_c * o * (1 - o))
        db_o = jnp.sum(dh_next * tanh_c * o * (1 - o), axis=0, keepdims=True)

        # Gradient of loss w.r.t. cell state
        d_c = dh_next * o * (1 - tanh_c ** 2) + dc_next

        # Gradient of loss w.r.t. input gate weights and biases
        dW_i = jnp.dot(concat_input.T, d_c * i * (1 - i))
        db_i = jnp.sum(d_c * i * (1 - i), axis=0, keepdims=True)

        # Gradient of loss w.r.t. candidate values weights and biases
        dW_c = jnp.dot(concat_input.T, d_c * (1 - c_hat ** 2))
        db_c = jnp.sum(d_c * (1 - c_hat ** 2), axis=0, keepdims=True)

        # Gradient of loss w.r.t. forget gate weights and biases
        dW_f = jnp.dot(concat_input.T, d_c * c_prev * (1 - f))
        db_f = jnp.sum(d_c * c_prev * (1 - f), axis=0, keepdims=True)

        # Compute gradients of loss w.r.t. concatenated input
        d_concat_input = jnp.dot(d_c, self.W_c.T)
        d_concat_input += jnp.dot(dW_o, self.W_o.T)
        d_concat_input += jnp.dot(dW_f, self.W_f.T)
        d_concat_input += jnp.dot(dW_i, self.W_i.T)

        # Extract gradients of loss w.r.t. previous hidden state and input
        d_h_prev = d_concat_input[:, :self.hidden_size]
        d_x = d_concat_input[:, self.hidden_size:]

        # Update weights and biases
        self.W_f -= learning_rate * dW_f
        self.b_f -= learning_rate * db_f

        self.W_i -= learning_rate * dW_i
        self.b_i -= learning_rate * db_i

        self.W_c -= learning_rate * dW_c
        self.b_c -= learning_rate * db_c

        self.W_o -= learning_rate * dW_o
        self.b_o -= learning_rate * db_o

        return d_h_prev, d_x