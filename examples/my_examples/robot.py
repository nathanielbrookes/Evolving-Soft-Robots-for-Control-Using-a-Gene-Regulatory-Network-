import numpy as np
import grn as GRN

import gym
from evogym import WorldObject

class Robot:
    def __init__(self, structure = None, controller = None):
        self.fitness = 0

        # Create initial robot
        if structure is None:
            # Loads 'speed_bot' fixed robot shape:
            robot_object = WorldObject.from_json("speed_bot.json")
            structure = robot_object.get_structure()

            """
            # Creates simple 2x3 robot made of horizontal actuators
            structure = np.array([[3, 3, 3], [3, 0, 3]])
            """

        self.structure = structure

        # Get sizes of observation space and action space:
        env = gym.make('DownStepper-v0', body=structure)
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        env.close()

        if controller is None:
            # Create GRN controller
            gene_count = self.observation_space + 32 + self.action_space
            controller = GRN.WatsonGRN(gene_count)
            controller.set_random_weights()

        self.controller = controller

        """
        # Calculate action space (by summing actuators)
        unique, counts = np.unique(self.structure, return_counts=True)
        occurrences = dict(zip(unique, counts))
        if 3 in occurrences:
            self.action_space += occurrences[3]
        if 4 in occurrences:
            self.action_space += occurrences[4]
        """

    def get_actuator_values(self):
        # Ends portion of phenotype represent the actuator values
        actuators = self.controller.get_phenotype()[-self.action_space:]

        # Scales actuators from [-1, 1] to [0.6, 1.6]
        #actuators_scaled = np.interp(actuators, (-1, 1), (0.6, 1.6))
        #print(actuators)

        return actuators

    def set_inputs(self, inputs):
        # Start portion of gene potentials are the inputs (sensors)

        for i in range(len(inputs)):
            self.controller.gene_potentials[i] = inputs[i]

    def set_fitness(self, fitness):
        self.fitness = fitness

    def step(self):
        self.controller.step()

    def reset(self):
        self.controller.reset()

