import gym
import evogym.envs

import random
import numpy as np
from robot import Robot
import os
import json
from ga import GeneticAlgorithm

if __name__ == '__main__':
    parameters = {
        'seed': 0,
        'shape': (5, 5),
        'pop_size': 250,
        'max_generations': 10000,
        'train_iterations': 500,
        'environment': 'Carrier-v0',
        'folder_name': 'co_evolve_test11'
    }
    
    seed = parameters['seed']
    random.seed(seed)
    np.random.seed(seed)

    start_gen = 0
    is_continuing = False

    # Create folder to store experiment data and results
    experiment_name = parameters['environment'] + '_' + parameters['folder_name']
    exp_path = 'experiment_data/' + experiment_name
    try:
        os.makedirs(exp_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "n":
            quit()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True

            # Retrieve parameters from metadata.json
            try:
                parameters_file = open(os.path.join(exp_path, 'metadata.json'))
                parameters = json.load(parameters_file)
            except:
                print('ERROR - metadata.json was not found!')

    # Store parameters
    if not is_continuing:
        parameters_file = os.path.join(exp_path, 'metadata.json')
        with open(parameters_file, 'w') as f:
            json.dump(parameters, f)

    if is_continuing:
        ga = GeneticAlgorithm(parameters['shape'], parameters['pop_size'], parameters['max_generations'],
                              parameters['train_iterations'], parameters['environment'], exp_path, start_gen, True)
    else:
        ga = GeneticAlgorithm(parameters['shape'], parameters['pop_size'], parameters['max_generations'],
                              parameters['train_iterations'], parameters['environment'], exp_path, start_gen, False)

    ga.start()

    """
    robot = Robot((5, 5))
    robot.develop()

    # Visualise robot
    env = gym.make('Walker-v0', body=robot.get_structure()[0])
    env.reset()

    t = 0
    while t < 500:
        # Maps robot actuator values to the actuators
        action = robot.get_actuator_values()

        ob, reward, done, info = env.step(action)

        env.render()

        t += 1

        if done:
            env.reset()

    env.close()
    """