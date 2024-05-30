import gym
import evogym.envs

import random
import numpy as np
from operator import attrgetter
from robot import Robot
import os
import json
from ga import GeneticAlgorithm

import sys

from grn import WatsonGRN


# Unit Test to check that new GRN.set_random_weights() works
def unit_test_1():
    grn = WatsonGRN(3)

    print(f'Gene Potentials: {grn.gene_potentials}')
    print(f'Interaction Matrix: {grn.interaction_matrix}')
    print()

    grn.set_random_weights()

    print(f'Gene Potentials after set_random_weights(): {grn.gene_potentials}')
    print(f'Interaction Matrix after set_random_weights(): {grn.interaction_matrix}')


# Unit Test to check that new GRN.mutate_weights() works
def unit_test_2():
    grn = WatsonGRN(3)

    print(f'Gene Potentials: {grn.gene_potentials}')
    print(f'Interaction Matrix: {grn.interaction_matrix}')
    print()

    grn.mutate_weights()

    print(f'Gene Potentials after mutate_weights(): {grn.gene_potentials}')
    print(f'Interaction Matrix after mutate_weights(): {grn.interaction_matrix}')


# Unit Test to check that Tournament Selection works
def unit_test_3():
    parents = []
    for i in range(10):
        robot = Robot('Walker-v0')
        robot.set_fitness(round(random.uniform(0, 10), 2))
        parents.append(robot)

        print(f'Parent {i} Fitness: {robot.fitness}')

    tournament = random.sample(parents, 2)

    print()
    print(f'Tournament Parent 1 Fitness: {tournament[0].fitness}')
    print(f'Tournament Parent 2 Fitness: {tournament[1].fitness}')
    print()

    parent = max(tournament, key=attrgetter('fitness'))

    print(f'Selected Parent Fitness: {parent.fitness}')


if __name__ == '__main__':
    parameters = {
        'seed': random.randint(0, 1E6),
        'shape': (5, 5),
        'pop_size': 5, # 100
        'max_generations': 2, # 200
        'environment': 'UpStepper-v0',
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
        
        print('FINISHED!')

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