import math
import random
import numpy as np

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
            if random.random() < 0.15:
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
        degradation_rate = -0.2

        # Calculate new gene potentials
        m = 0
        for i in range(self.gene_count):
            sum_of_activities = 0
            for j in range(self.gene_count):
                sum_of_activities += (self.interaction_matrix[m] * self.gene_potentials[j])
                m += 1

            self.phenotype[i] = (self.gene_potentials[i] + rate*math.tanh(sum_of_activities) + degradation_rate*self.gene_potentials[i]) * 5/9

            if (self.phenotype[i] < -1):
                self.phenotype[i] = -1
            elif (self.phenotype[i] > 1):
                self.phenotype[i] = 1

        self.gene_potentials = self.phenotype.copy()