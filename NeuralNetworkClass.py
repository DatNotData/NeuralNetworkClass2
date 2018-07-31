import numpy as np


class NeuralNetwork:
    def __init__(self, input_layer, output_layer, neuron):
        self.input_layer = input_layer  # input
        self.output_layer = output_layer  # expected output for given imput
        self.neuron = neuron  # neuron layout
        self.weight = [2 * np.random.random((self.neuron[i][0], self.neuron[i][1])) - 1 for i in range(self.neuron.shape[0])]  # initialize weight with random values

    @staticmethod
    def activation(x):  # activation function for our neural network
        return 1 / (1 + np.exp(-x))  # sigmoid

    @staticmethod
    def derivative(x):
        return x * (1 - x)  # sigmoid

    @staticmethod
    def get_dec_to_bin(dec, pad):
        b = bin(dec)[2:]
        return [int(i) for i in ('0' * (pad - len(b))) + b]

    def forward_propagation(self, input_layer):  # create all the necessary layers with their neurons with the appropriate values
        input_layer = [input_layer]
        for i in range(self.neuron.shape[0]):
            input_layer.append(self.activation(np.dot(input_layer[i], self.weight[i])))
        return input_layer

    def backward_propagation(self, layer):  # get the delta of each layer
        delta = [(self.output_layer - layer[self.neuron.shape[0]]) * self.derivative(layer[self.neuron.shape[0]])]  # get layers' delta and store them in an array
        for i in range(self.neuron.shape[0] - 1, 0, -1):
            delta.insert(0, delta[0].dot(self.weight[i].T) * self.derivative(layer[i]))
        return delta

    def gradient_descent(self, layer, delta):  # update weight array so that we reduce error
        return [self.weight[i] + layer[i].T.dot(delta[i]) for i in range(self.neuron.shape[0])]

    def train(self, cycles):
        for counter in range(cycles):  # train a number of times
            layer = self.forward_propagation(self.input_layer)
            delta = self.backward_propagation(layer)  # get the error
            self.weight = self.gradient_descent(layer, delta)

            if counter % 10000 == 0:  # debug : print average error
                print(np.mean(np.abs(self.output_layer - layer[-1])))
        print("Training completed \n\n\n")

    def get_output_layer(self, input_layer):
        return self.forward_propagation(input_layer)[-1]

    def get_rounded_output_layer(self, input_layer):
        return [i>0.5 for i in self.get_output_layer(input_layer)]

    def get_output_sum(self, input_layer):
        sum = 0
        for bit in [int(round(i)) for i in self.get_output_layer(input_layer)]:
            sum = (sum << 1) | bit
        return sum
