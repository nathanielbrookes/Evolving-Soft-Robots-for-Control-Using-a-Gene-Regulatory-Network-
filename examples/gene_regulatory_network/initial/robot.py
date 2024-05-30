import math
import numpy as np

import grn as GRN

import gym
from evogym import WorldObject
import evogym.envs

class Robot:
    def __init__(self, environment, controller = None):
        self.environment = environment
        self.structure = self.get_structure()
        
        # Get sizes of observation space and action space:
        env = gym.make(self.environment, body=self.structure[0])
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        env.close()
        
        if controller is None:
            # Create GRN controller
            gene_count = self.observation_space + 32 + self.action_space
            controller = GRN.WatsonGRN(gene_count)
            controller.set_random_weights()
        self.controller = controller

        self.robot_controller = None

        self.fitness = 0

    def instantiate_controller(self):
        self.robot_controller = RobotController(self)

    def get_structure(self):
        # Loads fixed robot shape:
        if self.environment == 'Carrier-v0':
            robot_object = WorldObject.from_json('carry_bot.json')
        else:
            robot_object = WorldObject.from_json('speed_bot.json')
            
        structure = (robot_object.get_structure(), robot_object.get_connections())

        return structure

    def get_actuator_values(self):
        action_values = self.robot_controller.step()
        return action_values
        
    def set_inputs(self, inputs):
        # Beginning portion of gene potentials are the inputs (sensors)
        for i in range(len(inputs)):
            self.robot_controller.controller.gene_potentials[i] = inputs[i]

    def set_fitness(self, fitness):
        self.fitness = fitness

    def reset(self):
        self.controller.reset()


class RobotController:
    def __init__(self, robot):
        self.robot = robot

        # Copy GRN controller from robot
        self.controller = GRN.WatsonGRN(robot.controller.gene_count)
        self.controller.interaction_matrix = robot.controller.interaction_matrix.copy()

    def step(self):
        # Steps controller
        self.controller.step()
        
        action_values = self.controller.get_phenotype()[-self.robot.action_space:]
        return action_values
        