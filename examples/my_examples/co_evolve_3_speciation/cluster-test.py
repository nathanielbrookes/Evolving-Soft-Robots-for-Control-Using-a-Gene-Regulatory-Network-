import gym
import evogym.envs

import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from operator import attrgetter
from robot import Robot, CalculateSimilarityMatrix
import os
import json
from ga import GeneticAlgorithm
import multiprocessing
import time
import random
from operator import attrgetter
from group_grn import SimJob, run_group_grn

if __name__ == '__main__':
    seed = random.randint(0, 1E6)
    random.seed(seed)
    np.random.seed(seed)
    
    population = []
    cluster_size = 5

    for i in range(25):
        robot = Robot((5, 5))
        population.append(robot)
        
    sim_jobs = []

    # Add current population as sim jobs (enables multiprocessing!!)
    for i, robot in enumerate(population):
        sim_jobs.append(SimJob(robot, 1000, 'Walker-v0', 'testing', i))

    process_count = multiprocessing.cpu_count()
    start = time.perf_counter()
    run_data = run_group_grn(sim_jobs, process_count)
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')

    population = []
    for result in run_data:
        population.append(result.robot)

    # Order population of robots according to fitness
    ordered_population = sorted(population, key=lambda x: x.fitness, reverse=True)

    for n, robot in enumerate(ordered_population, start=1):
        print(str(n) + ') ' + str(robot.fitness))
        
    ordered_population_no_extremes = []
            
    for robot in ordered_population:
        if robot.fitness > -100:
            ordered_population_no_extremes.append(robot)

    print('DONE!')

    # Get similarity matrix
    similarity_matrix = CalculateSimilarityMatrix(ordered_population_no_extremes)

    # Perform Hierarchical Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=cluster_size, affinity='precomputed', linkage='complete', distance_threshold=None)
    class_labels = agg.fit_predict(similarity_matrix)

    groups = {}
    for i in range(cluster_size):
        groups[i] = []

    # Copy robot into its group
    for i in range(len(class_labels)):
        group = class_labels[i]
        groups[group].append(ordered_population_no_extremes[i])
        
    print(class_labels)

    # Select 2 random groups for cross-group parent selection
    group_one, group_two = random.sample(list(groups), 2)

    # Select random robot from group one for first parent (tournament size = 5)
    group_one_candidates = random.sample(groups[group_one], min(5, len(groups[group_one])))
    parent_one = max(group_one_candidates, key=attrgetter('fitness'))

    # Select random robot from group two for second parent (tournament size = 5)
    group_two_candidates = random.sample(groups[group_two], min(5, len(groups[group_two])))
    parent_two = max(group_one_candidates, key=attrgetter('fitness'))

    exit()