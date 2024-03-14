import gym
from evogym.utils import get_full_connectivity

import evogym.envs
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from evogym import sample_robot

import os
import json
from ga import GeneticAlgorithm

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    parameters = {
        'pop_size': 25,
        'max_generations': 10000,
        'train_iterations': 500,
        'environment': 'Walker-v0',
        'folder_name': 'test11'
    }

    # Create folder to store experiment data and results
    experiment_name = parameters['environment'] + '_' + parameters['folder_name']
    exp_path = 'experiment_data/' + experiment_name
    try:
        os.makedirs(exp_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Override? (y/n): ", end="")
        ans = input()
        if ans.lower() == "y":
            pass
        else:
            quit()

    print(parameters)

    # Store parameters
    parameters_file = os.path.join(exp_path, 'metadata.json')
    with open(parameters_file, 'w') as f:
        json.dump(parameters, f)

    ga = GeneticAlgorithm(parameters['pop_size'], parameters['max_generations'], parameters['train_iterations'], parameters['environment'], exp_path)
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