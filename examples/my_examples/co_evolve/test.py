import gym
import evogym.envs

import random
import numpy as np
from robot import Robot
import os
import json
from ga import GeneticAlgorithm

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    parameters = {
        'shape': (10, 10),
        'pop_size': 25,
        'max_generations': 10000,
        'train_iterations': 500,
        'environment': 'Walker-v0',
        'folder_name': 'co_evolve_test3'
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

    ga = GeneticAlgorithm(parameters['shape'], parameters['pop_size'], parameters['max_generations'], parameters['train_iterations'], parameters['environment'], exp_path)
    ga.start()