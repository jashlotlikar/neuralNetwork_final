import numpy as np


class Layer:
    def __init__(self, input_size, output_size, layer_number):
        self.ZL = None
        self.AL = None
        self.layer_number = layer_number

        self.WL = np.random.randn(input_size, output_size) * 0.01
        self.BL = np.zeros((output_size, 1))

    def ZL(self, inputs):
        self.ZL = np.dot(inputs.T, self.WL).T + self.BL
    def AL(self):
        self.AL = self.Sigmoid(self.ZL)

    def forward(self, inputs):
        self.ZL = np.dot(inputs.T, self.WL).T + self.BL
        self.AL = self.Sigmoid(self.ZL)

    def load_parameters(self):
        filename = f"layer_parameters_{self.layer_number}.npz"
        self.WL = np.load(filename)["weights"]
        self.BL = np.load(filename)["biases"]

    def backward(self, base_gradient):
        dAl_dZL = self.Sigmoid_derivative()
        dZL_dAL_1 = self.WL
        return np.dot(dZL_dAL_1, base_gradient * dAl_dZL)
    
    def weight_gradient(self, base_gradient, inputs):
        dAl_dZL = self.Sigmoid_derivative()
        dZL_dWL = inputs
        return np.dot(dZL_dWL, (base_gradient * dAl_dZL).T)
    
    def bias_gradient(self, base_gradient):
        dAl_dZL = self.Sigmoid_derivative()
        return base_gradient * dAl_dZL

    def update_parameters(self,base_gradient, learning_rate):
        self.WL -= learning_rate * self.weight_gradient(base_gradient, self.inputs)
        self.BL -= learning_rate * self.bias_gradient(base_gradient)
        filename = f"layer_parameters_{self.layer_number}.npz"
        np.savez(filename, weights=self.WL, biases=self.BL)

    def Sigmoid(self, inp):
        return 1 / (1 + np.exp(-inp))
    def Sigmoid_derivative(self):
        return self.AL * (1 - self.AL)

    

