import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from robot import Robot
import grn as GRN

import gym

if __name__ == '__main__':
    environment = 'Walker-v0'
    exp_name = 'Walker-v0_test3'
    generation = 233
    robot = 0
    train_iterations = 500

    exp_path = os.path.join('experiment_data', exp_name, f'generation_{generation}')

    if not os.path.exists(exp_path):
        print('ERROR: Could not find experiment!')
        exit()
    else:
        controller_path = os.path.join(exp_path, 'controller', f'{robot}.npz')
        structure_path = os.path.join(exp_path, 'structure', f'{robot}.npz')

        if not os.path.exists(controller_path) or not os.path.exists(structure_path):
            print('ERROR: Could not find experiment structure and controller files!')
            exit()
        else:
            # Load structure data
            structure_data = np.load(structure_path)
            structure = []
            for key, value in structure_data.items():
                structure.append(value)
            structure = tuple(structure)
            structure, connections = structure

            # Load controller data
            controller_data = np.load(controller_path)
            controller = []
            for key, value in controller_data.items():
                controller.append(value)
            controller = tuple(controller)
            gene_count, interaction_matrix = controller

            robot_controller = GRN.WatsonGRN(gene_count)
            robot_controller.interaction_matrix = interaction_matrix

            robot = Robot(environment, [structure, connections], robot_controller)

            list_actuators = []
            for n in range(robot.action_space):
                list_actuators.append([])

            total_reward = 0

            # Visualise robot
            env = gym.make(environment, body=robot.structure[0])
            env.reset()
            t = 0
            while t < train_iterations:
                # Step robot
                robot.step()

                # Maps robot actuator values to the actuators
                action = robot.get_actuator_values()

                ob, reward, done, info = env.step(action)

                # Map observations to the inputs
                robot.set_inputs(ob)

                env.render()

                print(f'Reward: {reward}')
                total_reward += reward

                for n in range(len(action)):
                    list_actuators[n].append(action[n])

                t += 1

                if done:
                    #env.reset()
                    print(f'Total Reward: {total_reward}')

            env.close()

            # Plot matrix of evolved regulatory interactions
            matrix_2d = np.zeros((gene_count, gene_count))

            for i in range(gene_count):
                for j in range(gene_count):
                    matrix_2d[i][j] = interaction_matrix[i * gene_count + j]

            cmap = LinearSegmentedColormap.from_list('', ['#000', '#FFF'])
            plt.matshow(matrix_2d, cmap=cmap, vmin=matrix_2d.min(), vmax=matrix_2d.max())
            plt.colorbar()
            plt.show()

            # Plot actuators
            plt.title("Line graph")
            plt.xlabel("Development steps")
            plt.ylabel("Actuator expression levels")

            for n in range(min(robot.action_space, 1000)):
                plt.plot(np.arange(1, train_iterations + 1), list_actuators[n], label="Gene Actuators",
                         color=np.random.rand(3, ))

            plt.show()
