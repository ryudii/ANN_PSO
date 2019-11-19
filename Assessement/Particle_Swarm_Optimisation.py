import random


def cost_function(position_vector, neural_network, dataset):
    sample_error_list = []
    for sample in dataset:
        neural_network.list_of_layers[0].update_inputs(sample[:-1])
        neural_network.replace_with_new_vector_and_update(position_vector)
        neural_network_try = neural_network.get_answer()
        sample_error_list.append(float(sample[-1]) - neural_network_try)
    return (1 / len(sample_error_list)) * sum(sample_error_list) * sum(sample_error_list)


def fill_position_with_random_values(dimension):
    position = []
    for i in range(0, dimension):
        position.append(random.uniform(-1, 1))
    return position


class Particle:
    def __init__(self, dimension):
        self.position = fill_position_with_random_values(dimension)
        self.velocity = fill_position_with_random_values(dimension)
        self.pos_best = self.position[:]
        self.err_best = -1
        self.err = -1

    def update_velocity(self, best_swarm_pos, nb_dimension):
        constant_inertia_weight = 0.5
        cognitive_constant = 1
        social_constant = 2
        for i in range(0, nb_dimension):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = cognitive_constant * r1 * (self.pos_best[i] - self.position[i])
            social_velocity = social_constant * r2 * (best_swarm_pos[i] - self.position[i])
            inertia_velocity = constant_inertia_weight * self.velocity[i]
            self.velocity[i] = cognitive_velocity + social_velocity + inertia_velocity

    def update_position(self, nb_dimension, bounds):
        for i in range(0, nb_dimension):
            self.position[i] = self.position[i] + self.velocity[i]
            if self.position[i] > bounds[1]:
                self.position[i] = bounds[1]
            if self.position[i] > bounds[0]:
                self.position[i] = bounds[0]


class Swarm:
    def __init__(self, nb_particles, dimension):
        self.best_swarm_err = -1
        self.best_swarm_pos = []
        self.particle_list = []
        for index in range(0, nb_particles):
            self.particle_list.append(Particle(dimension))


def particle_swarm_optimisation(settings, neural_network, dataset):
    dimension = len(neural_network.get_vector_for_pso())
    nb_particles = settings[2]
    maxiter = settings[3]
    bounds = settings[4]
    swarm = Swarm(nb_particles, dimension)
    global_loop_turn = 0
    while global_loop_turn < maxiter:
        #if global_loop_turn % 10 == 0:
        #    print(str(global_loop_turn) + "%")
        for index in range(0, nb_particles):
            swarm.particle_list[index].err = cost_function(swarm.particle_list[index].position, neural_network, dataset)
            if swarm.particle_list[index].err_best > swarm.particle_list[index].err:
                swarm.particle_list[index].err_best = swarm.particle_list[index].err
                swarm.particle_list[index].pos_best = swarm.particle_list[index].position
            if swarm.best_swarm_err > swarm.particle_list[index].err or swarm.best_swarm_err == -1:
                swarm.best_swarm_pos = swarm.particle_list[index].position
                swarm.best_swarm_err = swarm.particle_list[index].err
        for index_2 in range(0, nb_particles):

            swarm.particle_list[index_2].update_velocity(swarm.best_swarm_pos, dimension)
            swarm.particle_list[index_2].update_position(dimension, bounds)

        global_loop_turn += 1
    return swarm.best_swarm_pos, swarm.best_swarm_err




