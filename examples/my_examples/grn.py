import math
import random
import numpy as np
import gym

class WatsonGRN:
    def __init__(self, gene_count):
        self.gene_count = gene_count
        self.gene_potentials = np.zeros(gene_count)
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
            elif random.random() < 0.1:
                # Mutate connection
                self.interaction_matrix[i] += random.uniform(-0.1, 0.1)

            if (self.interaction_matrix[i] < -1):
                self.interaction_matrix[i] = -1
            elif (self.interaction_matrix[i] > 1):
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

            self.phenotype[i] = (self.gene_potentials[i] + rate*math.tanh(sum_of_activities) - self.gene_potentials[i]*degradation_rate) * 0.55
            # * 5/9 ?

            if (self.phenotype[i] < -1):
                self.phenotype[i] = -1
            elif (self.phenotype[i] > 1):
                self.phenotype[i] = 1

        self.gene_potentials = self.phenotype.copy()

    def reset(self):
        self.gene_potentials = np.zeros(self.gene_count)
        self.phenotype = np.zeros(self.gene_count)


def RunGRN(robot, train_iters, env):
    # print(f'Structure: {robot.structure}')
    # print(f'Controller: {robot.controller}')

    env = gym.make(env, body=robot.structure)
    env.reset()

    current_fitness = 0

    t = 0
    while t < train_iters:
        # Step robot
        robot.step()

        # Maps robot actuator values to the actuators
        action = robot.get_actuator_values()

        ob, reward, done, info = env.step(action)

        # Map observations to the inputs
        robot.set_inputs(ob)

        # Update current fitness
        current_fitness = reward

        # env.render()
        t += 1

        if done:
            env.reset()

    env.close()

    # Return fitness of GRN robot:
    fitness = current_fitness
    return fitness
