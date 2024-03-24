import gym

import os
import numpy as np
import multiprocessing
import time
import random
from operator import attrgetter
from sklearn.cluster import AgglomerativeClustering

from group_grn import SimJob, run_group_grn
from robot import Robot, CalculateSimilarityMatrix
import grn as GRN


class GeneticAlgorithm:
    def __init__(self, shape, pop_size, generations, train_iterations, environment, experiment_dir, start_gen, is_continuing):
        self.shape = shape
        self.pop_size = pop_size
        self.generations = generations
        self.train_iterations = train_iterations
        self.environment = environment
        self.experiment_dir = experiment_dir
        self.start_gen = start_gen
        self.is_continuing = is_continuing

        self.population = []

        if not self.is_continuing:
            # Create an initial population of random robots:
            # Robots are defined by structure of connections (design) and GRN (controller)
            for i in range(pop_size):
                robot = Robot(self.shape)
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

            """
                Elitism Strategy
                
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
            """

            """
            My Strategy 
            
            new_population = [ordered_population[0]]
            
            # Replace remaining spaces with offspring
            while len(new_population) < self.pop_size:
                if random.random() < 0.5:
                    # Select random robot for parent (tournament size = 5)
                    parents = random.sample(ordered_population, 5)
                    parent = max(parents, key=attrgetter('fitness'))

                    # Create new robot with mutated child controller:
                    child_grn = GRN.WatsonGRN(parent.controller.gene_count)
                    child_grn.interaction_matrix = parent.controller.interaction_matrix.copy()
                    child_grn.mutate_weights()
                    child_robot = Robot(parent.container_shape, child_grn)

                    # Add mutated child to the new population
                    new_population.append(child_robot)

                else:
                    # Select random robots for parents (tournament size = 5)
                    parents = random.sample(ordered_population, 5)
                    parent_one = max(parents, key=attrgetter('fitness'))
                    parents = random.sample(ordered_population, 5)
                    parent_two = max(parents, key=attrgetter('fitness'))

                    # Perform crossover to get child controller
                    child_grn = GRN.CrossoverGRN(parent_one.controller, parent_two.controller)
                    child_grn.mutate_weights()

                    # Create new robot with mutated child controller:
                    child_robot = Robot(parent_one.container_shape, child_grn)

                    # Add mutated child to the new population
                    new_population.append(child_robot)
            """

            cluster_size = 10
           
            ordered_population_no_extremes = []
            
            for robot in ordered_population:
                if robot.fitness > -100:
                    ordered_population_no_extremes.append(robot)

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
                
            for i in range(cluster_size):
                print(f'Group {i}: {len(groups[i])}')

            new_population = []
            # Replace remaining spaces with offspring
            while len(new_population) < self.pop_size:
                if random.random() < 0.8:
                    if random.random() < 0.5:
                        # Simple Crossover

                        # Select random robots for parents (tournament size = 8)
                        parents = random.sample(ordered_population_no_extremes, 8)
                        parent_one = max(parents, key=attrgetter('fitness'))
                        parents = random.sample(ordered_population_no_extremes, 8)
                        parent_two = max(parents, key=attrgetter('fitness'))

                        # Perform crossover to get child controller
                        child_grn = GRN.CrossoverGRN(parent_one.controller, parent_two.controller)
                        child_grn.mutate_weights()

                        # Create new robot with mutated child controller:
                        child_robot = Robot(parent_one.container_shape, child_grn)

                        # Add mutated child to the new population
                        new_population.append(child_robot)

                    else:
                        # Cross-group Crossover

                        # Select random robots for parents:

                        # Select 2 random groups for cross-group parent selection
                        group_one, group_two = random.sample(list(groups), 2)

                        # Select random robot from group one for first parent (tournament size = 5)
                        group_one_candidates = random.sample(groups[group_one], min(5, len(groups[group_one])))
                        parent_one = max(group_one_candidates, key=attrgetter('fitness'))

                        # Select random robot from group two for second parent (tournament size = 5)
                        group_two_candidates = random.sample(groups[group_two], min(5, len(groups[group_two])))
                        parent_two = max(group_two_candidates, key=attrgetter('fitness'))

                        # Perform crossover to get child controller
                        child_grn = GRN.CrossoverGRN(parent_one.controller, parent_two.controller)
                        child_grn.mutate_weights()

                        # Create new robot with mutated child controller:
                        child_robot = Robot(parent_one.container_shape, child_grn)

                        # Add mutated child to the new population
                        new_population.append(child_robot)

                else:
                    # Mutation

                    # Select random robot for parent (tournament size = 8)
                    parents = random.sample(ordered_population, 8)
                    parent = max(parents, key=attrgetter('fitness'))

                    # Create new robot with mutated child controller:
                    child_grn = GRN.WatsonGRN(parent.controller.gene_count)
                    child_grn.interaction_matrix = parent.controller.interaction_matrix.copy()
                    child_grn.mutate_weights()
                    child_robot = Robot(parent.container_shape, child_grn)

                    # Add mutated child to the new population
                    new_population.append(child_robot)

            # Update population
            self.population = new_population

            g += 1


            """new_population = []

            # Replace remaining spaces with offspring
            while len(new_population) < self.pop_size:
                if random.random() < 0.8:
                    # Crossover

                    # Select random robots for parents (tournament size = 8)
                    parents = random.sample(ordered_population, 8)
                    parent_one = max(parents, key=attrgetter('fitness'))
                    parents = random.sample(ordered_population, 8)
                    parent_two = max(parents, key=attrgetter('fitness'))

                    # Perform crossover to get child controller
                    child_grn = GRN.CrossoverGRN(parent_one.controller, parent_two.controller)
                    child_grn.mutate_weights()

                    # Create new robot with mutated child controller:
                    child_robot = Robot(parent_one.container_shape, child_grn)

                    # Add mutated child to the new population
                    new_population.append(child_robot)

                else:
                    # Mutation

                    # Select random robot for parent (tournament size = 8)
                    parents = random.sample(ordered_population, 8)
                    parent = max(parents, key=attrgetter('fitness'))

                    # Create new robot with mutated child controller:
                    child_grn = GRN.WatsonGRN(parent.controller.gene_count)
                    child_grn.interaction_matrix = parent.controller.interaction_matrix.copy()
                    child_grn.mutate_weights()
                    child_robot = Robot(parent.container_shape, child_grn)

                    # Add mutated child to the new population
                    new_population.append(child_robot)

            # Update population
            self.population = new_population

            g += 1"""


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