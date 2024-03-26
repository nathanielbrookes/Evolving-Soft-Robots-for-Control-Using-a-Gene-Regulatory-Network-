import math
import numpy as np

import grn as GRN
from evogym import get_full_connectivity


class Robot:
    def __init__(self, container_shape, controller = None):
        if controller is None:
            # Create GRN controller
            gene_count = 32
            controller = GRN.WatsonGRN(gene_count)
            controller.set_random_weights()

        self.controller = controller

        self.observation_space = 4 # 4 morphogen inputs
        # 4 morphogen outputs + divide threshold and 2 axes + 4 values to represent gene state
        self.action_space = 4 + 3 + 4

        self.container_shape = container_shape
        self.voxels = {}
        self.actuators = {}

        self.fitness = 0

    def develop(self):
        # Add starting voxel at the centre
        width, height = self.container_shape
        centre = (math.floor(width / 2), math.floor(height / 2))
        self.voxels[centre] = RobotVoxel(centre, self)

        for i in range(300):
            # Simulate all voxels
            for key in list(self.voxels.keys()):
                voxel = self.voxels[key]
                voxel.simulate()

    def get_structure(self):
        body = np.zeros(self.container_shape, dtype=int)

        for key in list(self.voxels.keys()):
            voxel = self.voxels[key]
            if not voxel.isMature:
                # Force cell to be mature
                voxel.isMature = True
                voxel.determine_fate()

            body[key] = voxel.fate+1

        return [body, get_full_connectivity(body)]

    def get_actuator_values(self):
        action_values = np.zeros(len(self.actuators))
        for i, (key, value) in enumerate(self.actuators.items()):
            actuator = value
            action = actuator.simulate()
            action_values[i] = action

        return action_values

    def set_fitness(self, fitness):
        self.fitness = fitness

    def reset(self):
        self.controller.reset()


class RobotVoxel:
    def __init__(self, location, robot):
        self.location = location
        self.robot = robot

        # Copy GRN controller from robot to each voxel
        self.controller = GRN.WatsonGRN(robot.controller.gene_count)
        self.controller.interaction_matrix = robot.controller.interaction_matrix.copy()

        self.isMature = False
        self.isActuator = False
        self.fate = None
        self.divide_attempts = 4

    def get_morphogen(self, direction):
        action_values = self.controller.get_phenotype()[-self.robot.action_space:]
        morphogen_values = action_values

        if direction == 'L':
            return morphogen_values[0]
        elif direction == 'U':
            return morphogen_values[1]
        elif direction == 'R':
            return morphogen_values[2]
        elif direction == 'D':
            return morphogen_values[3]

    def simulate(self):
        # Determine and set morphogen inputs for the controller
        col, row = self.location
        leftMorphogen = upMorphogen = rightMorphogen = downMorphogen = 0
        if (col-1, row) in self.robot.voxels:
            leftMorphogen = self.robot.voxels[(col-1, row)].get_morphogen('R')
        if (col, row-1) in self.robot.voxels:
            upMorphogen = self.robot.voxels[(col, row-1)].get_morphogen('D')
        if (col+1, row) in self.robot.voxels:
            rightMorphogen = self.robot.voxels[(col+1, row)].get_morphogen('L')
        if (col, row+1) in self.robot.voxels:
            downMorphogen = self.robot.voxels[(col, row+1)].get_morphogen('U')

        self.controller.gene_potentials[0] = leftMorphogen
        self.controller.gene_potentials[1] = upMorphogen
        self.controller.gene_potentials[2] = rightMorphogen
        self.controller.gene_potentials[3] = downMorphogen

        # Step controller
        self.controller.step()

        if self.isMature:
            if self.isActuator:
                action_value = self.controller.get_phenotype()[-self.robot.action_space:][10:][0]
                return action_value
        else:
            # Get output values (divide threshold, direction & axis)
            action_values = self.controller.get_phenotype()[-self.robot.action_space:][4:]
            threshold_value = action_values[0]
            direction_value = action_values[1]
            axis_value = action_values[2]

            """threshold_value = 1
            direction_value = random.uniform(-1, 1)
            axis_value = random.uniform(-1, 1)"""

            divide_threshold = 0.65
            # Check if voxel can divide
            if threshold_value > divide_threshold or threshold_value < -divide_threshold:
                self.divide_attempts -= 1
                # Attempt to divide voxel
                if axis_value < 0:
                    # Attempt to divide voxel horizontally
                    if direction_value < 0:
                        # Attempt to divide voxel to left
                        if col-1 >= 0 and (col-1, row) not in self.robot.voxels:
                            self.robot.voxels[(col-1, row)] = RobotVoxel((col-1, row), self.robot)
                    else:
                        # Attempt to divide voxel to right
                        if col+1 <= self.robot.container_shape[0]-1 and (col+1, row) not in self.robot.voxels:
                            self.robot.voxels[(col+1, row)] = RobotVoxel((col+1, row), self.robot)
                else:
                    # Attempt to divide vertically
                    if direction_value < 0:
                        # Attempt to divide voxel up
                        if row-1 >= 0 and (col, row-1) not in self.robot.voxels:
                            self.robot.voxels[(col, row-1)] = RobotVoxel((col, row-1), self.robot)
                    else:
                        # Attempt to divide voxel down
                        if row+1 <= self.robot.container_shape[1]-1 and (col, row+1) not in self.robot.voxels:
                            self.robot.voxels[(col, row+1)] = RobotVoxel((col, row+1), self.robot)

            if self.divide_attempts <= 0:
                # Voxel becomes mature -> Now define voxel fate
                self.isMature = True
                self.determine_fate()

    def determine_fate(self):
        action_values = self.controller.get_phenotype()[-self.robot.action_space:][4:]
        fate_values = action_values[3:]

        self.fate = max(range(len(fate_values)), key=fate_values.__getitem__)

        # Add actuators voxels
        if self.fate == 2 or self.fate == 3:
            self.isActuator = True
            self.robot.actuators[self.location] = self
