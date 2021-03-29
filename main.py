from math import exp
from random import seed
from random import random


def init_network(n_inputs, n_neurons_hidden_layer, n_outputs):
    network = list()
    hidden_layer = [{"weights": [random() for _ in range(n_inputs + 1)]} for _ in range(n_neurons_hidden_layer)]
    network.append(hidden_layer)
    output_layer = [{"weights": [random() for _ in range(n_neurons_hidden_layer + 1)]} for _ in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    return sum([weights[i] * inputs[i] for i in range(len(weights) - 1)]) + weights[-1]


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def transfer_derivative(output):
    return output * (1.0 - output)


def forward_propagate(network, input_data):
    inputs = input_data
    for layer in network:
        new_input = []
        for neuron in layer:
            activation_value = activate(neuron["weights"], inputs)
            neuron["output"] = transfer(activation_value)
            new_input.append(neuron["output"])
        inputs = new_input
    return inputs


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i == len(network) - 1:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron["output"])
        else:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron["weights"][j] * neuron["delta"])
                errors.append(error)
        for j in range(len(layer)):
            neuron = layer[j]
            print(errors[j], transfer_derivative(neuron["output"]))
            neuron["delta"] = errors[j] * transfer_derivative(neuron["output"])


h1 = {"output": 0.593269992, "weights": [0.15, 0.20]}
h2 = {"output": 0.596884378, "weights": [0.30, 0.25]}
o1 = {"output": 0.75136507, "weights": [0.40, 0.45]}
o2 = {"output": 0.772928465, "weights": [0.50, 0.55]}

network = [[h1, h2], [o1, o2]]
expected = [0.01, 0.99]
backward_propagate_error(network, expected)
for layer in network:
    print(layer)
