import gym
import math
import random
import numpy as np
import os
import robot as bot

class WatsonGRN:
    def __init__(self, gene_count):
        self.gene_count = gene_count
        self.gene_potentials = np.zeros(gene_count)
        for i in range(self.gene_count):
            self.gene_potentials[i] = 1
        self.interaction_matrix = np.zeros(gene_count * gene_count)
        self.phenotype = np.zeros(gene_count)

    def get_phenotype(self):
        return self.phenotype

    def set_random_weights(self):
        for i in range(self.gene_count):
            self.gene_potentials[i] = random.uniform(-0.1, 0.1)

        for i in range(self.gene_count*self.gene_count):
            if random.random() < 0.1:
                self.interaction_matrix[i] = random.uniform(-1, 1)

    def mutate_weights(self):
        for i in range(self.gene_count*self.gene_count):
            if random.random() < 0.01:
                # Remove connection
                self.interaction_matrix[i] = 0
            elif random.random() < 0.05:
                # Mutate connection
                self.interaction_matrix[i] += random.uniform(-0.1, 0.1)

            if self.interaction_matrix[i] < -1:
                self.interaction_matrix[i] = -1
            elif self.interaction_matrix[i] > 1:
                self.interaction_matrix[i] = 1


    def step(self):
        rate = 1
        degradation_rate = 0.02

        # Calculate new gene potentials
        m = 0
        for i in range(self.gene_count):
            sum_of_activities = 0
            for j in range(self.gene_count):
                sum_of_activities += (self.interaction_matrix[m] * self.gene_potentials[j])
                m += 1

            self.phenotype[i] = (self.gene_potentials[i] + rate*math.tanh(sum_of_activities) - self.gene_potentials[i]*degradation_rate)

            if self.phenotype[i] < -1:
                self.phenotype[i] = -1
            elif self.phenotype[i] > 1:
                self.phenotype[i] = 1

        self.gene_potentials = self.phenotype.copy()

    def reset(self):
        self.gene_potentials = np.zeros(self.gene_count)
        self.phenotype = np.zeros(self.gene_count)


# Crossover operator for GRN controller
# Requires both GRNs to have the same gene count!
# Based on Karl Sims (1994) crossover operator for Mating Directed Graphs
def CrossoverGRN(parent_one, parent_two):
    if parent_two.gene_count != parent_one.gene_count:
        print('PARENT GRNs DO NOT HAVE SAME LENGTHS!')
        return False

    child_gene_count = parent_one.gene_count
    child_interaction_matrix = np.zeros(child_gene_count * child_gene_count)

    # Seperate interaction matrix into gene connection arrays for each parent
    parent_one_genes = []
    parent_two_genes = []
    for i in range(0, parent_one.gene_count):
        parent_one_genes.append(np.take(parent_one.interaction_matrix, range(i*parent_one.gene_count, (i+1)*parent_one.gene_count)))
    for i in range(0, parent_two.gene_count):
        parent_two_genes.append(np.take(parent_two.interaction_matrix, range(i*parent_two.gene_count, (i+1)*parent_two.gene_count)))

    # Select random crossover point
    crossover_point = random.randint(0, child_gene_count)

    # Perform crossover
    for i in range(0, child_gene_count):
        if i < crossover_point:
            # Use genes from parent one
            gene_connections = parent_one_genes[i]
        else:
            # Use genes from parent two
            gene_connections = parent_two_genes[i]

        for j in range(0, child_gene_count):
            child_interaction_matrix[(i * child_gene_count) + j] = gene_connections[j]

    # Return new GRN with crossover
    child_grn = WatsonGRN(child_gene_count)
    child_grn.interaction_matrix = child_interaction_matrix.copy()
    return child_grn


def RunGRN(robot, train_iters, env, generation_path, index):
    # Develop robot
    robot.develop()

    if len(robot.actuators) == 0:
        print('Robot has no actuators! - Returning -100 for fitness')
        return -100

    # Save robot GRN controller (defined by gene_count and interaction_matrix)
    controller_path = os.path.join(generation_path, 'controller')
    temp_path = os.path.join(controller_path, f'{index}.npz')
    np.savez(temp_path, robot.controller.gene_count, robot.controller.interaction_matrix)

    # Save robot structure and connections array
    structure_path = os.path.join(generation_path, 'structure')
    temp_path = os.path.join(structure_path, f'{index}.npz')
    structure = robot.get_structure()
    np.savez(temp_path, structure[0], structure[1])

    env = gym.make(env, body=robot.get_structure()[0])
    env.reset()

    current_fitness = 0
    total_fitness = 0

    t = 0
    while t < train_iters:
        # Maps robot actuator values to the actuators
        action = robot.get_actuator_values()

        ob, reward, done, info = env.step(action)

        # Map observations to the inputs
        #robot.set_inputs(ob)

        # Update current fitness
        current_fitness = reward
        total_fitness += current_fitness

        # env.render()
        t += 1

        if done:
            env.reset()

    env.close()

    # Return fitness of GRN robot:
    fitness = total_fitness
    return fitness