import gym

import os
import numpy as np
import multiprocessing
import time
import random

from group_grn import SimJob, run_group_grn
from robot import Robot
import grn as GRN


class GeneticAlgorithm:
    def __init__(self, pop_size, generations, train_iterations, environment, experiment_dir):
        self.pop_size = pop_size
        self.generations = generations
        self.train_iterations = train_iterations
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

                sim_jobs.append(SimJob(robot, self.train_iterations, self.environment))

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

            # Add best ~ 50% of robots to new population
            survivors = int(0.5 * self.pop_size)
            new_population = ordered_population[:survivors]

            # Replace remaining spaces with offspring
            while len(new_population) < self.pop_size:
                # Select best robot for parent
                parent_one = ordered_population.pop(0)

                # Select random robot for parent
                parent_two = random.sample(ordered_population, 1)[0]

                # Perform crossover to get child controller
                child_grn = GRN.CrossoverGRN(parent_one.controller, parent_two.controller)

                # Create new robot with mutated child controller:
                child_robot = Robot(self.environment)
                child_robot.controller = child_grn
                child_robot.controller.mutate_weights()

                # Add mutated child to the new population
                new_population.append(child_robot)

            # Update population
            self.population = new_population

            g += 1
