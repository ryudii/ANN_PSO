import sys
import Neural_Network as nn
import Particle_Swarm_Optimisation as pso
import time


# function to transform the .txt who is already split by lines into a usable dataset
# First we try to split by space and if it didnt split anything we split by tabulation.
# This work for all the given files
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


# function to setup the neural network using the settings we created.
# neural_network contain list of layers who contains list of neurons
# the dataset parameter is here to fill the input layer.
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


# Function you modify to change all the hyperparameters
# There is no need think about the input layer here.
def choose_settings():
    number_of_neurons_per_layer = [2, 2, 2, 2, 2]
    number_of_layers = len(number_of_neurons_per_layer)
    number_of_particles = 100
    number_of_pso_iteration = 100
    bounds = [-20, 20]
    settings = [number_of_layers, number_of_neurons_per_layer, number_of_particles, number_of_pso_iteration, bounds]
    return settings


# Function to import the dataset from the path
# With readlines() we get a list of string. Each line is already splitted like this : ["line 1", "line2",...., line100]
def import_dataset(path):
    f = open(path, "r")
    text_by_line = f.readlines()
    dataset = get_dataset_from_file(text_by_line)
    f.close()
    return dataset


# This function is the last call of the main loop. It's here you write what you have to before going to the next file.
# Here i am incrementing a list of the best vector with his cost and the name of the function we worked on.
# e.g. [vector]             0.001                 1in_cubic.txt
#         |                    |                        |
#         V                    V
#      list of vector    list of best cost          list of name
# then i print the cost we've got for instant result at the screen without more infos
def use_result_of_this_loop(vector, cost, name, vector_list, cost_list, name_list):
    cost_list.append(cost)
    vector_list.append(vector)
    name_list.append(name)
    print(name + " : " + str(cost))
    return vector_list, cost_list, name_list


# function to know wich activation function are used
def retrieve_activation_functions_vector_only(vector, number_of_inputs):
    index = number_of_inputs - 1
    activation_function_vector = []
    while index != len(vector) - 1:
        if index % 2 == 1:
            activation_function_vector.append(abs(int((vector[index + number_of_inputs] * 100) % 5)))
        index += 1
    return activation_function_vector


# This function is the last call of the whole program.
# It's here we take care of keeping all the infos we need to do stats and
# have a better understanding of what happen with hyperparameters changes
def use_final_results(vector_list, cost_list, name_list, settings):
    f = open("new_results.txt", "a")
    index = 0
    for vector in vector_list:
        af_list = retrieve_activation_functions_vector_only(vector, int(name_list[index][0]))
        f.write(name_list[index][:-4] + "," + str(sum(settings[1])) + "," + str(settings[0]) + "," +
                str(settings[2]) + "," + str(settings[3]) + "," + str(settings[4][1]) + "," + str(settings[4][0]) + "," +
                str(cost_list[index]) + "," +
                str(af_list.count(0)) + "," + str(af_list.count(1)) + "," + str(af_list.count(2)) + "," +
                str(af_list.count(3)) + "," + str(af_list.count(4)) + "\n")
        index += 1

# index help us go through each file we have to work on.
# cost_list vector_list name_list are used to get the infos from the pso. It will host all the results
# settings will host all the hyperparameters of the ANN and PSO. More infos above choose_settings() function
if __name__ == '__main__':
    for x in range(0, 100):
        print(x)
        begin_time = time.time()
        index = 1
        cost_list = []
        vector_list = []
        name_list = []
        settings = choose_settings()
        while index != len(sys.argv):
            dataset = import_dataset(sys.argv[index])
            neural_network = set_up_neural_network(dataset, settings)
            vector, cost = pso.particle_swarm_optimisation(settings, neural_network, dataset)
            vector_list, cost_list, name_list = use_result_of_this_loop(vector, cost, sys.argv[index][5:], vector_list, cost_list, name_list)
            index += 1
        end_time = time.time()
        use_final_results(vector_list, cost_list, name_list, settings)
        print("It took : " + str(end_time - begin_time) + " seconds")

