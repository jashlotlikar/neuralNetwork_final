import network as net
import numpy as np

def construct_network(layer_sizes, input_size, output_size):
    layers = []
    for i in range(len(layer_sizes) - 1):
        if layer_sizes[i] <= 0 or layer_sizes[i + 1] <= 0:
            raise ValueError("Layer sizes must be positive integers.")
        if i == 0:
            layer = net.Layer(input_size, layer_sizes[i + 1], i)
        elif i != len(layer_sizes) - 1:
            layer = net.Layer(layer_sizes[i], layer_sizes[i + 1], i)
        else:
            layer = net.Layer(layer_sizes[i], output_size, i)
        layers.append(layer)
    return layers

def set_layer_parameters(layers):
    for i, layer in enumerate(layers):
        layer.load_parameters()

def forward_pass(layers, inputs):
    for layer in layers:
        layer.forward(inputs)
        inputs = layer.AL
    return inputs

def backward_pass(layers, X, Y):

    base_gradient = 2 * (forward_pass(layers, X) - Y)

    for layer in reversed(layers):
        layer.update_parameters(base_gradient, learning_rate=0.01)
        base_gradient = layer.backward(base_gradient)
    return base_gradient

def calculate_loss(X, Y, layers):
    predictions = forward_pass(layers, X)
    return np.mean((predictions - Y) ** 2)

def train_network(layers, X, Y, epochs):
    for epoch in range(epochs):
        backward_pass(layers, X, Y)
        loss = calculate_loss(X, Y, layers)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")