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