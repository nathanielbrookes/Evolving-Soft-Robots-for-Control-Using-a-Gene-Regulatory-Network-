import gym
import evogym.envs

import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from operator import attrgetter
from robot import Robot, RobotVoxel
from grn import WatsonGRN
import os
import json
from ga import GeneticAlgorithm

import sys


# Unit Test to check that RobotVoxel.determine_fate() works
def unit_test_1():
    robot = Robot((5, 5))

    voxel = RobotVoxel((0, 0), robot)
    fate_values = voxel.controller.phenotype[-robot.action_space:][4:][3:] = [0, 0, 1, 0]

    print(f'Voxel Fate Genes: {fate_values}')
    print()

    voxel.determine_fate()

    print(f'Voxel Fate = Material Type {voxel.fate}')


# Unit Test to check that Robot.develop() works
def unit_test_2():
    found = False
    while not found:
        robot = Robot((5, 5))

        robot.develop()

        if len(robot.actuators) == 0:
            print('Robot has no actuators!')
        else:
            found = True

    env = gym.make('Walker-v0', body=robot.get_structure()[0])
    env.reset()

    t = 0
    finished = False
    while not finished:
        # Maps robot actuator values to the actuators
        action = robot.get_actuator_values()

        ob, reward, done, info = env.step(action)

        env.render()

        if done:
            env.reset()
            finished = True
        else:
            t += 1

    env.close()


if __name__ == '__main__':
    parameters = {
        'seed': random.randint(0, 1E6),
        'shape': (5, 5),
        'pop_size': 5, # 200
        'max_generations': 3, # 200
        'environment': 'Walker-v0',
        'folder_name': 'test'
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
        ga = GeneticAlgorithm(parameters['shape'], parameters['pop_size'], parameters['max_generations'], parameters['environment'], exp_path, start_gen, True)
    else:
        ga = GeneticAlgorithm(parameters['shape'], parameters['pop_size'], parameters['max_generations'], parameters['environment'], exp_path, start_gen, False)

    print(f"Seed = {parameters['seed']}")
    ga.start()

    """
        while True:
        robot = Robot((10, 10))
        robot.develop()

        # Visualise robot
        try:
            env = gym.make('Walker-v0', body=robot.get_structure()[0])
            env.reset()

            t = 0
            while t < 100:
                # Maps robot actuator values to the actuators
                action = robot.get_actuator_values()

                ob, reward, done, info = env.step(action)

                env.render()

                t += 1

                if done:
                    env.reset()

            env.close()

        except:
            pass

    exit()
    """

    """
    robots = []
    cluster_size = 5

    for i in range(15):
        robot = Robot((5, 5))
        robot.develop()

        # TBD - Evaluate Robots!

        robots.append(robot)

    print('DONE!')

    # Get similarity matrix
    similarity_matrix = CalculateSimilarityMatrix(robots)

    # Perform Hierarchical Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=cluster_size, affinity='precomputed', linkage='single')
    class_labels = agg.fit_predict(similarity_matrix)

    groups = {}
    for i in range(cluster_size):
        groups[i] = []

    # Copy robot into its group
    for i in range(len(class_labels)):
        group = class_labels[i]
        groups[group].append(robots[i])

    # Select 2 random groups for cross-group parent selection
    group_one, group_two = random.sample(list(groups), 2)

    # Select random robot from group one for first parent (tournament size = 5)
    group_one_candidates = random.sample(groups[group_one], min(5, len(groups[group_one])))
    parent_one = max(group_one_candidates, key=attrgetter('fitness'))

    # Select random robot from group two for second parent (tournament size = 5)
    group_two_candidates = random.sample(groups[group_two], min(5, len(groups[group_one])))
    parent_two = max(group_one_candidates, key=attrgetter('fitness'))

    print(parent_one)
    print(parent_two)

    exit()
    """