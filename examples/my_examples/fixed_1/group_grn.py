# This is a Python script
# Purpose is to handle group operations (running batches of robots in parallel) using multiprocessing

import mp_group as mp
from grn import RunGRN

# SimJob class has been adapted from EvoGym for use in a Genetic Algorithm
class SimJob:
    def __init__(self, robot, env):
        self.robot = robot
        self.env = env

    def get_data(self):
        return {'robot': self.robot, 'envs': self.env}

# RunData class has been adapted from EvoGym for use in a Genetic Algorithm
class RunData:
    def __init__(self, robot, env):
        self.robot = robot
        self.env = env

    def set_fitness(self, fitness):
        self.robot.set_fitness(fitness)
        self.robot.reset()

def run_group_grn(sim_jobs, process_count):
    run_data = []
    group = mp.Group()

    for job in sim_jobs:
        robot = job.robot
        env = job.env

        run_data.append(RunData(robot, env))

        grn_args = (robot, env)
        group.add_job(RunGRN, grn_args, callback=run_data[-1].set_fitness)

    # Runs process_count number of processes
    group.run_jobs(process_count)

    return run_data
