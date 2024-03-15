import numpy as np
from lstm import LSTM

class MultiLayerLSTM:
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialize weights and biases for each layer
        self.layers = []
        for _ in range(num_layers):
            layer = LSTM(input_size, hidden_size, output_size)
            self.layers.append(layer)
            input_size = hidden_size  # Set input size of next layer to hidden size of current layer

    def forward(self, x):
        hidden_states = []
        cell_states = []
        output = None
        for layer in self.layers:
            output, hidden_state, cell_state = layer.forward(x)
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)
            x = hidden_state  # Pass the hidden state of current layer as input to next layer
        return output, hidden_states, cell_states

    def backward(self, d_output, hidden_states, cell_states, learning_rate):
        d_prev_hidden_state = None
        for layer, hidden_state, cell_state in zip(reversed(self.layers), reversed(hidden_states), reversed(cell_states)):
            d_prev_hidden_state, _ = layer.backward(d_output, d_prev_hidden_state, cell_state, learning_rate)
            d_output = None  # We only need to compute gradients with respect to the output of the last layer
