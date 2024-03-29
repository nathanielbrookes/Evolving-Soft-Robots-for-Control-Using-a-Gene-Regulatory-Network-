import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

if __name__ == '__main__':
    exp_name = 'test_ga_Hurdler-v0'

    exp_path = os.path.join('saved_data', exp_name)

    if not os.path.exists(exp_path):
        print('ERROR: Could not find experiment!')
        exit()
    else:
        results = []

        # Loop through generations to collect results
        g = 0
        while os.path.exists(os.path.join(exp_path, f'generation_{g}', 'output.txt')):
            output_file = os.path.join(exp_path, f'generation_{g}', 'output.txt')

            best_value = None
            with open(output_file) as f:
                first_line = f.readline().strip('\n')
                values = first_line.split('\t\t')
                best_value = float(values[1])

            # Take best fitness from each generation
            results.append(best_value)

            g += 1

        # Plot the best fitness across generations:
        plt.title("Fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")

        plt.plot(np.arange(0, len(results)), results, label="Best Fitness")
        plt.legend()

        """
        # Fit the trend line
        z = np.polyfit(np.arange(0, len(results)), results, 5)
        p = np.poly1d(z)
        plt.plot(np.arange(0, len(results)), p(np.arange(0, len(results))))
        """

        plt.show()