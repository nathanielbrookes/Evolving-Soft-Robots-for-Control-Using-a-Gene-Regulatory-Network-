import imageio
import os
import numpy as np
from robot import Robot
import grn as GRN
import gym
import evogym.envs


if __name__ == '__main__':
    environment = 'Walker-v0'
    exp_name = 'Walker-v0_co_evolve_test8'
    container_shape = (5, 5)
    generation = 271

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
            robot_indices = output[:, 1]

            for i in range(len(robot_indices)):
                index = int(robot_indices[i])

                controller_path = os.path.join(exp_path, 'controller', f'{index}.npz')
                structure_path = os.path.join(exp_path, 'structure', f'{index}.npz')

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

                    robot = Robot(container_shape, robot_controller)
                    robot.develop()

                    list_actuators = []
                    for n in range(len(robot.actuators)):
                        list_actuators.append([])

                    total_reward = 0

                    # Visualise robot
                    env = gym.make(environment, body=structure)
                    env.reset()

                    # Maps robot actuator values to the actuators
                    action = robot.get_actuator_values()

                    ob, reward, done, info = env.step(action)
                    img = env.render(mode='img')
                    env.close()

                    # Save image
                    images_dir = os.path.join(exp_path, 'images')
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)

                    image_path = os.path.join(images_dir, f'{i}.png')
                    imageio.imwrite(image_path, img)
