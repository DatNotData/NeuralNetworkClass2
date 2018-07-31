import numpy as np

from NeuralNetworkClass import NeuralNetwork

data_file = open('data.csv', "r")
data = data_file.read()
data_file.close()
data = data.split('\n')

number_input_values = 2

input_list = []
output_list = []

for line in data:
    line = line.split(',')
    line = [float(i) for i in line]
    input_list.append(line[0:number_input_values])
    output_list.append(line[number_input_values:])

input_list = np.array(input_list)
output_list = np.array(output_list)

print(input_list)

np.random.seed(1)  # set seed so that debug is easier

neuron = np.array([[len(input_list[0]), 4], [4, 8], [8, 4], [4, len(output_list[0])]])  # array of neurons

neural_network = NeuralNetwork(input_list, output_list, neuron)  # create the neural network
neural_network.train(50000)

for i in input_list:
    print(i, '  :  ', neural_network.get_output_layer(i))

