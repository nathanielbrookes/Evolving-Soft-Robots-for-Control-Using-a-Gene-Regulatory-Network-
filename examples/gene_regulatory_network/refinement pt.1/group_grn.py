# This is a Python script
# Purpose is to handle group operations (running batches of robots in parallel) using multiprocessing

import os
import numpy as np
import mp_group as mp
from grn import RunGRN

# SimJob class has been adapted from EvoGym for use in a Genetic Algorithm
class SimJob:
    def __init__(self, robot, env, generation_path, index):
        self.robot = robot
        self.env = env
        self.generation_path = generation_path
        self.index = index

    def get_data(self):
        return {'robot': self.robot, 'envs': self.env, 'generation_path': self.generation_path, 'index': self.index}

# RunData class has been adapted from EvoGym for use in a Genetic Algorithm
class RunData:
    def __init__(self, robot, env, generation_path, index):
        self.robot = robot
        self.env = env
        self.generation_path = generation_path
        self.index = index

    def set_fitness(self, fitness):
        self.robot.reset()
        self.robot.set_fitness(fitness)
        
        # Set body from loaded structure data
        structure_path = os.path.join(self.generation_path, 'structure', f'{self.index}.npz')
        structure_data = np.load(structure_path)
        structure = []
        for key, value in structure_data.items():
            structure.append(value)
        structure = tuple(structure)
        body, connections = structure
        self.robot.body = body
        self.robot.index = self.index
        
        
def run_group_grn(sim_jobs, process_count):
    run_data = []
    group = mp.Group()

    for job in sim_jobs:
        robot = job.robot
        env = job.env
        generation_path = job.generation_path
        index = job.index

        run_data.append(RunData(robot, env, generation_path, index))

        grn_args = (robot, env, generation_path, index)
        group.add_job(RunGRN, grn_args, callback=run_data[-1].set_fitness)

    # Runs process_count number of processes
    group.run_jobs(process_count)

    return run_data
