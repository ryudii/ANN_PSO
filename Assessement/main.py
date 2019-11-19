import sys
import Neural_Network as nn
import Particle_Swarm_Optimisation as pso


def get_dataset_from_file(text_by_line):
    dataset = []
    for line in text_by_line:
        line = line[:-1]
        line_in_list = line.split(" ")
        if len(line_in_list) == 1:
            line_in_list = line.split("\t")
        line_in_list = list(filter(None, line_in_list))
        dataset.append(line_in_list)
    return dataset


def set_up_neural_network(dataset, settings):
    nb_layers = settings[0]
    nb_neurons_per_layer = settings[1]
    my_neural_network = nn.NeuralNetwork()
    input_layer = nn.InputLayer(dataset[0])
    my_neural_network.list_of_layers.append(input_layer)
    for i in range(0, nb_layers):
        if i == 0:
            last_layer_value = input_layer.compute_layer_value()
        else:
            last_layer_value = my_neural_network.list_of_layers[i - 1].compute_layer_value()
        my_neural_network.list_of_layers.append(nn.Layer(nb_neurons_per_layer[i], last_layer_value))
    return my_neural_network


def choose_settings():
    number_of_layers = 3
    number_of_neurons_per_layer = [2, 2, 2]
    number_of_particles = 100
    number_of_pso_iteration = 100
    bounds = [-2, 2]
    if len(number_of_neurons_per_layer) != number_of_layers:
        print("Error in settings, number of layers should be the same as the len of number of neurons per layer list")
    settings = [number_of_layers, number_of_neurons_per_layer, number_of_particles, number_of_pso_iteration, bounds]
    return settings


if __name__ == '__main__':
    index = 1
    while index != len(sys.argv):
        path = sys.argv[index]
        f = open(path, "r")
        text_by_line = f.readlines()
        dataset = get_dataset_from_file(text_by_line)
        f.close()
        settings = choose_settings()
        cost_list = []
        vector_list = []
        neural_network = set_up_neural_network(dataset, settings)
        vector, cost = pso.particle_swarm_optimisation(settings, neural_network, dataset)
        cost_list.append(cost)
        vector_list.append(vector)
        print(path[5:] + " : " + str(cost))
        index += 1

