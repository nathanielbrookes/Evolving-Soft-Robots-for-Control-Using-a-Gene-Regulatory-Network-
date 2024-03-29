import gym

import os
import math
import numpy as np
import multiprocessing
import time
import random
from operator import attrgetter

from group_grn import SimJob, run_group_grn
from robot import Robot
import grn as GRN


class GeneticAlgorithm:
    def __init__(self, pop_size, generations, environment, experiment_dir):
        self.pop_size = pop_size
        self.generations = generations
        self.environment = environment
        self.experiment_dir = experiment_dir

        self.population = []

        # Get sizes of observation space and action space:
        env = gym.make(environment, body=Robot(self.environment).structure[0])
        print(f'Observation Space = {env.observation_space.shape[0]}')
        print(f'Action Space = {env.action_space.shape[0]}')
        env.close()

        # Create an initial population of robots:
        # Robots are defined by structure of connections (design) and GRN (controller)
        for i in range(pop_size):
            robot = Robot(self.environment)

            self.population.append(robot)

    def start(self):
        g = 0
        while g < self.generations:
            print('Generation ' + str(g))

            # Create subfolder for each generation
            generation_path = f'{self.experiment_dir}/generation_{g}'
            if not os.path.exists(generation_path):
                os.makedirs(generation_path)

            # Create subfolders for each generation structure and controller
            structure_path = os.path.join(generation_path, 'structure')
            if not os.path.exists(structure_path):
                os.makedirs(structure_path)
            controller_path = os.path.join(generation_path, 'controller')
            if not os.path.exists(controller_path):
                os.makedirs(controller_path)

            sim_jobs = []

            # Add current population as sim jobs (enables multiprocessing!!)
            for i, robot in enumerate(self.population):
                # Save robot structure and connections array
                temp_path = os.path.join(structure_path, f'{i}.npz')
                np.savez(temp_path, robot.structure[0], robot.structure[1])

                # Save robot GRN controller (defined by gene_count and interaction_matrix)
                temp_path = os.path.join(controller_path, f'{i}.npz')
                np.savez(temp_path, robot.controller.gene_count, robot.controller.interaction_matrix)

                sim_jobs.append(SimJob(robot, self.environment))

            process_count = multiprocessing.cpu_count()
            start = time.perf_counter()
            run_data = run_group_grn(sim_jobs, process_count)
            finish = time.perf_counter()
            print(f'Finished in {round(finish - start, 2)} second(s)')

            self.population = []
            for result in run_data:
                self.population.append(result.robot)

            # Order population of robots according to fitness
            ordered_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

            for n, robot in enumerate(ordered_population, start=1):
                print(str(n) + ') ' + str(robot.fitness))

            # Save fitness data to file
            output = []
            for robot in ordered_population:
                output.append(robot.fitness)

            a = np.asarray(output)
            np.savetxt(os.path.join(generation_path, 'output.csv'), a, delimiter=",")

            # Visualise best robot
            """env = gym.make(self.environment, body=ordered_population[0].structure[0])
            env.reset()
            t = 0
            while t < self.train_iterations:
                # Step robot
                ordered_population[0].step()

                # Maps robot actuator values to the actuators
                action = ordered_population[0].get_actuator_values()

                ob, reward, done, info = env.step(action)

                # Map observations to the inputs
                ordered_population[0].set_inputs(ob)

                env.render()

                t += 1

                if done:
                    env.reset()

            env.close()"""


            """
            # Strategy 1 & 2 
            
            new_population = []
            
            # Select the best quarter of robots as parents
            parents = ordered_population[:math.ceil(0.25 * self.pop_size)]
            
            # Computer the element-wise mean µ of parents
            grn_matrices = [parent.controller.interaction_matrix for parent in parents]
            parents_mean = np.mean(grn_matrices)
            
            # Add the best robot to new population
            elite_grn = GRN.WatsonGRN(parents[0].controller.gene_count)
            elite_grn.interaction_matrix = parents[0].controller.interaction_matrix
            elite_robot = Robot(self.environment)
            elite_robot.controller = elite_grn
            new_population.append(elite_robot)

            # Replace remaining spaces with offspring
            i = 0
            while len(new_population) < self.pop_size:
                # Create new child robot GRN by adding guassian noise to the parents mean:
                child_grn = GRN.WatsonGRN(parents[0].controller.gene_count)
                child_grn.interaction_matrix = parents_mean + np.random.normal(0, 0.35, parents[0].controller.gene_count * parents[0].controller.gene_count)

                # Create new child robot:
                child_robot = Robot(self.environment)
                child_robot.controller = child_grn

                # Add mutated child to the new population
                new_population.append(child_robot)
                
                i += 1

            # Update population
            self.population = new_population
            """
            
            # Strategy 3
            
            new_population = []
            
            # Select the best quarter of robots as parents
            parents = ordered_population[:math.ceil(0.25 * self.pop_size)]
            
            # Add the best robot to new population
            elite_grn = GRN.WatsonGRN(parents[0].controller.gene_count)
            elite_grn.interaction_matrix = parents[0].controller.interaction_matrix.copy()
            elite_robot = Robot(self.environment)
            elite_robot.controller = elite_grn
            new_population.append(elite_robot)

            # Replace remaining spaces with offspring
            i = 0
            while len(new_population) < self.pop_size:
                # Selecting parent using tournament selection (size = 2)
                tournament = random.sample(parents, 2)
                parent = max(tournament, key=attrgetter('fitness'))
                
                # Create new child robot GRN by adding guassian noise to the parents mean:
                child_grn = GRN.WatsonGRN(parent.controller.gene_count)
                child_grn.interaction_matrix = parent.controller.interaction_matrix.copy()

                # Create new robot with mutated child controller:
                child_robot = Robot(self.environment)
                child_robot.controller = child_grn
                child_robot.controller.mutate_weights()

                # Add mutated child to the new population
                new_population.append(child_robot)
                
                i += 1

            # Update population
            self.population = new_population

            g += 1
