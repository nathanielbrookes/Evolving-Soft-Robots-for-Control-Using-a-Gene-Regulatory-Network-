import gym

import multiprocessing
import time
from group_grn import SimJob, run_group_grn
from robot import Robot
import grn as GRN


class GeneticAlgorithm:
    def __init__(self, pop_size, generations, train_iterations):
        self.pop_size = pop_size
        self.generations = generations
        self.train_iterations = train_iterations

        self.population = []

        # Get sizes of observation space and action space:
        env = gym.make('DownStepper-v0', body=Robot().structure)
        print(env.observation_space.shape)
        print(env.action_space.shape)
        env.close()

        # Create an initial population of robots:
        # Robots are defined by structure of connections (design) and GRN (controller)
        for i in range(pop_size):
            robot = Robot()

            self.population.append(robot)

    def start(self):
        g = 0
        while g < self.generations:
            print('Generation ' + str(g))

            sim_jobs = []

            # Add current population as sim jobs (enables multiprocessing!!)
            for robot in self.population:
                sim_jobs.append(SimJob(robot, self.train_iterations, 'DownStepper-v0'))

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

            # Visualise best robot
            """env = gym.make('DownStepper-v0', body=ordered_population[0].structure)
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

            survivors = int(0.2*self.pop_size)

            # Add best ~ 20% of robots to new population
            new_population = ordered_population[:survivors]

            # Replace remaining spaces with offspring
            while len(new_population) < self.pop_size:
                # Select best robot for parent
                parent_robot = ordered_population.pop(0)

                # Create new robot with mutated parent controller:
                child_robot = Robot()
                child_robot.controller = GRN.WatsonGRN(parent_robot.controller.gene_count)
                child_robot.controller.interaction_matrix = parent_robot.controller.interaction_matrix.copy()
                child_robot.controller.mutate_weights()

                # Add mutated child to the new population
                new_population.append(child_robot)

            # Update population
            self.population = new_population

            g += 1
