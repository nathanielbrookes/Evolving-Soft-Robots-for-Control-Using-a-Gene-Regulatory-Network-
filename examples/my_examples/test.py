import gym
from evogym.utils import get_full_connectivity

import evogym.envs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from evogym import sample_robot

from ga import GeneticAlgorithm

if __name__ == '__main__':
    ga = GeneticAlgorithm(25, 5000, 1000)
    ga.start()

# connections = get_full_connectivity(body)

    """final_interaction_matrix = grn.interaction_matrix
    if T % 200 == 0:
        # Plot matrix of evolved regulatory interactions
        matrix_2d = np.zeros((gene_count, gene_count))

        for i in range(gene_count):
            for j in range(gene_count):
                matrix_2d[i][j] = final_interaction_matrix[i * gene_count + j]

        cmap = LinearSegmentedColormap.from_list('', ['#000', '#FFF'])
        plt.matshow(matrix_2d, cmap=cmap, vmin=matrix_2d.min(), vmax=matrix_2d.max())
        plt.colorbar()
        plt.show()"""