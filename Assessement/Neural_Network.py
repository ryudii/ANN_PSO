import activation_functions as af
import random


def activation_function(activation_function_choice, value):
    activation_function_choice = abs(int((activation_function_choice * 100) % 5))
    if activation_function_choice == 0:
        return af.null_activation_function()
    elif activation_function_choice == 1:
        return af.sigmoid_activation_function(value)
    elif activation_function_choice == 2:
        return af.hyperbolic_tangent_activation_function(value)
    elif activation_function_choice == 3:
        return af.cosine_activation_function(value)
    elif activation_function_choice == 4:
        return af.gaussian_activation_function(value)
    else:
        print("ERROR, you call for activation function : " + str(activation_function_choice))


class Input:
    def __init__(self, value):
        self.value = value
        self.weight = (random.randint(0, 100) - 1) / 100
        self.activation_fct = None

    def update_value(self, last_layer_value):
        self.value = activation_function(self.activation_fct, last_layer_value)


class InputLayer:
    def __init__(self, input_list):
        input_list = input_list[:-1]
        self.list_of_neurons = []
        for input in input_list:
            self.list_of_neurons.append(Input(input))

    def compute_layer_value(self):
        layer_value = 0
        for input in self.list_of_neurons:
            layer_value += float(input.value) * input.weight
        return layer_value

    def update_inputs(self, new_inputs):
        index = 0
        for input in self.list_of_neurons:
            input.value = new_inputs[index]
            index += 1


class Layer:
    def __init__(self, nb_of_neurons, value_of_last_layer):
        self.list_of_neurons = []
        for x in range(0, nb_of_neurons):
            self.list_of_neurons.append(Neuron(value_of_last_layer))

    def compute_layer_value(self):
        layer_value = 0
        for neurons in self.list_of_neurons:
            layer_value += float(neurons.value) * neurons.weight
        return layer_value


class Neuron:
    def __init__(self, last_layer_value):
        self.activation_fct = (random.randint(0, 100) - 1) / 100
        self.value = activation_function(self.activation_fct, last_layer_value)
        self.weight = (random.randint(0, 100) - 1) / 100

    def update_value(self, last_layer_value):
        self.value = activation_function(self.activation_fct, last_layer_value)


class NeuralNetwork:
    def __init__(self):
        self.list_of_layers = []

    def get_answer(self):
        if len(self.list_of_layers) > 0:
            return self.list_of_layers[-1].compute_layer_value()

    def get_vector_for_pso(self):
        vector_for_pso = []
        for layer in self.list_of_layers:
            for neuron in layer.list_of_neurons:
                vector_for_pso.append(neuron.weight)
                if neuron.activation_fct is not None:
                    vector_for_pso.append(neuron.activation_fct)
        return vector_for_pso

    def replace_with_new_vector_and_update(self, vector):
        index = 0
        layer_nb = 0
        for layer in self.list_of_layers:
            for neuron in layer.list_of_neurons:
                neuron.weight = vector[index]
                index += 1
                if neuron.activation_fct is not None:
                    neuron.activation_fct = vector[index]
                    index += 1
                    neuron.update_value(self.list_of_layers[layer_nb - 1].compute_layer_value())
            layer_nb += 1
