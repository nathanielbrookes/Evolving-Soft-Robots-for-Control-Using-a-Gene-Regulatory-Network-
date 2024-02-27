import gym
from evogym.utils import get_full_connectivity

import evogym.envs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from evogym import sample_robot

import grn as GRN

gene_count = 64

if __name__ == '__main__':
    # Create GRN
    grn = GRN.WatsonGRN(gene_count)
    grn.set_random_weights()

    # Creates simple 2x3 robot made of horizontal actuators
    body = np.array([[3, 3, 3], [3, 0, 3]])
    connections = get_full_connectivity(body)

    env = gym.make('UpStepper-v0', body=body)
    env.reset()

    T = 0
    while True:
        # Step GRN model
        grn.step()

        if T % 50 == 0:
            print("MUTATE!")
            grn.mutate_weights()

            env.reset()

        # Map 5 phenotypes to the actuators
        action = grn.get_phenotype()[-5:]
        #print(action)

        ob, reward, done, info = env.step(action)

        # Map observations to the inputs
        print(len(ob))
        for i in range(len(ob)):
            grn.gene_potentials[i] = ob[i]

        env.render()

        final_interaction_matrix = grn.interaction_matrix

        if T % 200 == 0:
            # Plot matrix of evolved regulatory interactions
            """matrix_2d = np.zeros((gene_count, gene_count))

            for i in range(gene_count):
                for j in range(gene_count):
                    matrix_2d[i][j] = final_interaction_matrix[i * gene_count + j]

            cmap = LinearSegmentedColormap.from_list('', ['#000', '#FFF'])
            plt.matshow(matrix_2d, cmap=cmap, vmin=matrix_2d.min(), vmax=matrix_2d.max())
            plt.colorbar()
            plt.show()"""


        T += 1

        if done:
            env.reset()

    env.close()