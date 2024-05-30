import gym

import os
import numpy as np
import multiprocessing
import time
import math
import random
from operator import attrgetter

from group_grn import SimJob, run_group_grn
from robot import Robot
import grn as GRN


class GeneticAlgorithm:
    def __init__(self, shape, pop_size, generations, environment, experiment_dir, start_gen, is_continuing):
        self.shape = shape
        self.pop_size = pop_size
        self.generations = generations
        self.environment = environment
        self.experiment_dir = experiment_dir
        self.start_gen = start_gen
        self.is_continuing = is_continuing

        self.population = []
        self.groups = {}

        if not self.is_continuing:
            # Create an initial population of random robots:
            # Robots are defined by structure of connections (design) and GRN (controller)
            for i in range(pop_size):
                robot = Robot(self.environment)
                self.population.append(robot)

        else:
            exp_path = os.path.join(experiment_dir, f'generation_{self.start_gen}')
            if not os.path.exists(exp_path):
                print('ERROR: Could not find experiment!')
                exit()
            else:
                population = []
                fitness_scores = []
                robot_indices = []
                # Load output data
                output_file = os.path.join(exp_path, 'output.csv')
                if not os.path.exists(output_file):
                    print('ERROR: Could not find experiment fitness scores!')
                    exit()
                else:
                    output = np.loadtxt(output_file, delimiter=',')
                    fitness_scores = output[:, 0]
                    robot_indices = output[:, 1]

                # Load population from existing generation to start next generation
                for i in range(self.pop_size):
                    controller_path = os.path.join(exp_path, 'controller', f'{i}.npz')
                    structure_path = os.path.join(exp_path, 'structure', f'{i}.npz')

                    if not os.path.exists(controller_path) or not os.path.exists(structure_path):
                        print('ERROR: Could not find experiment structure and controller files!')
                        exit()
                    else:
                        # Load controller data
                        controller_data = np.load(controller_path)
                        controller = []
                        for key, value in controller_data.items():
                            controller.append(value)
                        controller = tuple(controller)
                        gene_count, interaction_matrix = controller
                        robot_controller = GRN.WatsonGRN(gene_count)
                        robot_controller.interaction_matrix = interaction_matrix

                        robot = Robot(self.shape, robot_controller)
                        
                        # Set body from loaded structure data
                        structure_data = np.load(structure_path)
                        structure = []
                        for key, value in structure_data.items():
                            structure.append(value)
                        structure = tuple(structure)
                        body, connections = structure
                        robot.body = body
                        
                        robot.index = robot_indices[i]
                        robot.set_fitness(fitness_scores[i])
                        population.append(robot)

                self.population = population

    def start(self):
        g = 0

        if self.is_continuing:
            g = self.start_gen

        while g < self.generations:
            if not self.is_continuing:

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
                    sim_jobs.append(SimJob(robot, self.environment, generation_path, i))

                process_count = multiprocessing.cpu_count()
                start = time.perf_counter()
                run_data = run_group_grn(sim_jobs, process_count)
                finish = time.perf_counter()

                print('Generation ' + str(g))
                print(f'Finished in {round(finish - start, 2)} second(s)')

                self.population = []
                for result in run_data:
                    self.population.append(result.robot)

                # Order population of robots according to fitness
                ordered_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

                for n, robot in enumerate(ordered_population, start=1):
                    print(f'{n}) {robot.fitness} [index {robot.index}]')

                # Save fitness data to file
                output = []
                for robot in ordered_population:
                    output.append([robot.fitness, int(robot.index)])

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

            else:
                print('Continuing Generation ' + str(g))
                ordered_population = self.population
                self.is_continuing = False
                
                
            ordered_population_no_extremes = []
            
            for robot in ordered_population:
                if robot.fitness > -100:
                    ordered_population_no_extremes.append(robot)


            # Strategy 1 & 2 - Application Stage
            
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

            
            """new_population = []
            
            # Select the best quarter of robots as parents
            parents = ordered_population[:math.ceil(0.25 * self.pop_size)]
            
            # Add the best robot to new population
            elite = ordered_population[0]
            elite_grn = GRN.WatsonGRN(elite.controller.gene_count)
            elite_grn.interaction_matrix = elite.controller.interaction_matrix.copy()
            elite_robot = Robot(elite.container_shape, elite_grn)
            new_population.append(elite_robot)

            # Replace remaining spaces with offspring
            while len(new_population) < self.pop_size:
                # Selecting parent using tournament selection (size = 2)
                tournament = random.sample(parents, 2)
                parent = max(tournament, key=attrgetter('fitness'))
                
                # Create new robot with mutated child controller:
                child_grn = GRN.WatsonGRN(parent.controller.gene_count)
                child_grn.interaction_matrix = parent.controller.interaction_matrix.copy()
                child_grn.mutate_weights()
                child_robot = Robot(parent.container_shape, child_grn)

                # Add mutated child to the new population
                new_population.append(child_robot)
                
            # Update population
            self.population = new_population"""

            g += 1