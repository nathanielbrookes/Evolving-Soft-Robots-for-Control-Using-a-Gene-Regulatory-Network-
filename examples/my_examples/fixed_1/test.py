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
    parameters = {
        'seed': 220074,
        'shape': (5, 5),
        'pop_size': 36,
        'max_generations': 10000,
        'environment': 'Walker-v0',
        'folder_name': 'test3'
    }
    
    seed = parameters['seed']
    random.seed(seed)
    np.random.seed(seed)

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

    ga = GeneticAlgorithm(parameters['pop_size'], parameters['max_generations'], parameters['environment'], exp_path)
    ga.start()