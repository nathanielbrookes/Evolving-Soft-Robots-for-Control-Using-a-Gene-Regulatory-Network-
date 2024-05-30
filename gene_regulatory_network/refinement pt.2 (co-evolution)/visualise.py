import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from robot import Robot
import grn as GRN

import imageio

import gym
import evogym.envs

if __name__ == '__main__':
    environment = 'Thrower-v0'
    exp_name = 'Thrower-v0_Coevolve_0'
    container_shape = (5, 5)
    generation = 41
    robot = 0

    exp_path = os.path.join('experiment_data', exp_name, f'generation_{generation}')

    if not os.path.exists(exp_path):
        print('ERROR: Could not find experiment!')
        exit()
    else:
        fitness_scores = []
        robot_indices = []
        # Load output data
        output_file = os.path.join(exp_path, 'output.csv')
        if not os.path.exists(output_file):
            print('ERROR: Could not find experiment output.csv file!')
            exit()
        else:
            output = np.loadtxt(output_file, delimiter=',')
            fitness_score = output[:, 0][robot]
            robot_index = int(output[:, 1][robot])
            
            print(robot_index)
            
            controller_path = os.path.join(exp_path, 'controller', f'{robot_index}.npz')
            structure_path = os.path.join(exp_path, 'structure', f'{robot_index}.npz')
        
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

            robot = Robot(container_shape, robot_controller)
            robot.develop()
            
            structure = robot.get_structure()

            list_actuators = []
            for n in range(len(robot.actuators)):
                list_actuators.append([])

            total_reward = 0

            print(robot.body)

            # Visualise robot
            env = gym.make(environment, body=robot.body)
            env.reset()
            t = 0
            finished = False
            
            images = []
            
            while not finished:
                # Maps robot actuator values to the actuators
                action = robot.get_actuator_values()

                ob, reward, done, info = env.step(action)

                # Map observations to the inputs
                #robot.set_inputs(ob)

                env.render()
                
                if t % 100 == 0:
                    images.append(env.render(mode = 'img'))

                print(f'Reward: {reward}')
                total_reward += reward

                for n in range(len(action)):
                    list_actuators[n].append(action[n])

                if done:
                    env.reset()
                    finished = True
                else:
                    t += 1

            env.close()
            
            images_dir = 'images'
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            for i in range(len(images)):
                img = images[i]
                image_path = os.path.join(images_dir, f'{environment}_{i*100}.png')
                imageio.imwrite(image_path, img)

            print(f'Total Reward: {total_reward}')
            print(f'Steps: {t}')

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

            for n in range(min(1000, len(robot.actuators))):
                plt.plot(np.arange(1, t + 2), list_actuators[n], label="Gene Actuators",
                         color=np.random.rand(3, ))

            plt.show()
