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
    def __init__(self, shape, pop_size, generations, train_iterations, environment, experiment_dir):
        self.shape = shape
        self.pop_size = pop_size
        self.generations = generations
        self.train_iterations = train_iterations
        self.environment = environment
        self.experiment_dir = experiment_dir

        self.population = []

        # Create an initial population of random robots:
        # Robots are defined by structure of connections (design) and GRN (controller)
        for i in range(pop_size):
            robot = Robot(self.shape)
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
                sim_jobs.append(SimJob(robot, self.train_iterations, self.environment, generation_path, i))

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
                child_grn.mutate_weights()

                # Create new robot with mutated child controller:
                child_robot = Robot(parent_one.container_shape, child_grn)

                # Add mutated child to the new population
                new_population.append(child_robot)

            # Update population
            self.population = new_population

            g += 1


"""max_shape = (10, 10)

    robot = Robot(max_shape)
    while len(robot.actuators) == 0:
        # print('Robot has no actuators! -- Regenerating')
        robot = Robot(max_shape)

    body = robot.get_structure()

    env = gym.make('Walker-v0', body=body)
    env.reset()

    T = 500
    t = 0
    while t < T:
        action_values = np.zeros(len(robot.actuators))
        for i, (key, value) in enumerate(robot.actuators.items()):
            actuator = value
            action = actuator.simulate()
            action_values[i] = action

        print(action_values)

        action = action_values
        ob, reward, done, info = env.step(action)
        env.render()

        t += 1

        if done:
            env.reset()

    env.close()"""