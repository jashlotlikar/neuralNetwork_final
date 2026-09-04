import network_runner as runner
import network as net
import numpy as np

def main():
    # Define the architecture of the neural network
    layer_sizes = [3, 5, 2]  # Example: 3 input features, 5 neurons in hidden layer, 2 output classes
    input_size = layer_sizes[0]
    output_size = layer_sizes[-1]

    # Construct the neural network
    layers = runner.construct_network(layer_sizes, input_size, output_size)

    runner.set_layer_parameters(layers)

    generate_sample_data(num_samples=100, input_size=input_size, output_size=output_size)
    # Generate some random training data
    X = np.load("sample_data.npz")["X"]  # 100 samples with 'input_size' features
    Y = np.load("sample_data.npz")["Y"]  # Corresponding random outputs

    # Train the neural network
    epochs = 1000
    runner.train_network(layers, X, Y, epochs)


def generate_sample_data(num_samples=100, input_size=3, output_size=2):
    np.random.seed(412)  # For reproducibility
    X = np.random.rand(input_size, num_samples)
    np.random.seed(42)  # For reproducibility
    Y = np.random.rand(output_size, num_samples)
    np.savez("sample_data.npz", X=X, Y=Y)

main()