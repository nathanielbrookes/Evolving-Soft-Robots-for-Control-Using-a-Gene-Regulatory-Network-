import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    exp_name = 'Walker-v0_test5'

    exp_path = os.path.join('experiment_data', exp_name)

    if not os.path.exists(exp_path):
        print('ERROR: Could not find experiment!')
        exit()
    else:
        results = []
        results_avg = []
        results_worst = []

        # Loop through generations to collect results
        g = 0
        while os.path.exists(os.path.join(exp_path, f'generation_{g}', 'output.csv')):
            output_file = os.path.join(exp_path, f'generation_{g}', 'output.csv')

            array = np.loadtxt(output_file, delimiter=',')
            # Take best fitness from each generation
            results.append(array[0])

            # Take average fitness from each generation
            results_avg.append(sum(array)/len(array))

            # Take worst fitness from each generation
            results_worst.append(array[-1])

            g += 1

        # Plot the best fitness across generations:
        plt.title("Fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")

        plt.plot(np.arange(0, len(results)), results, label="Best Fitness")
        plt.plot(np.arange(0, len(results_avg)), results_avg, label="Average Fitness")
        plt.plot(np.arange(0, len(results_worst)), results_worst, label="Worst Fitness")

        plt.legend()
        plt.show()
