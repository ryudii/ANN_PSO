import activation_functions as af
import random


# Function who use an activation function on a neuron value.
# The first line allow us to get a choice from any number the pso will choose.
# %5 to get a number between 0 and 4 and abs() to get rid of the minus if needed.
# After understanding the PSO choice it call an activation function stored in the activation_functions.py
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


# Input class, look a lot like the Neuron class but i keep them separated to avoid mistakes
class Input:
    def __init__(self, value):
        self.value = value
        self.weight = (random.randint(0, 100) - 1) / 100
        self.activation_fct = None

    def update_value(self, last_layer_value):
        self.value = activation_function(self.activation_fct, last_layer_value)


# InputLayer class, look a lot like the Layer class but i keep them separated to avoid mistakes
# It take care of the update we have to do to input during the cost function fo the pso
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


# Layer class, it contain a list of his neurons
# the layer can sum up all of his neurons to give it to the next layer.
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


# Neuron class, with an activation function by neuron, a value and a weight.
class Neuron:
    def __init__(self, last_layer_value):
        self.activation_fct = (random.randint(0, 100) - 1) / 100
        self.value = activation_function(self.activation_fct, last_layer_value)
        self.weight = (random.randint(0, 100) - 1) / 100

    def update_value(self, last_layer_value):
        self.value = activation_function(self.activation_fct, last_layer_value)


# The neural network class, contain a list of layer.
# The pso will call each of his function.
# get_answer() is called to get a result with the current configuration
# get_vector_for_pso() is called to represent the solution in the pso
# replace_with_new_vector_and_update is called to change the weights and activation functions of the neural network

class NeuralNetwork:
    def __init__(self):
        self.mean_result_list = []
        self.mean_desired_result_list = []
        self.mse = []
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
