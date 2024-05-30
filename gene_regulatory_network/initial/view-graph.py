import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

if __name__ == '__main__':
    show_improvements_only = False

    exp_name = 'Walker-v0_InitialTest_0'

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
            fitness_scores = array[:, 0]
            indices = array[:, 1]
                   
            # Take best fitness from each generation    
            if g > 0 and fitness_scores[0] <= results[-1] and show_improvements_only:
                results.append(results[-1])
            else: 
                results.append(fitness_scores[0])

            # Take average fitness from each generation
            results_avg.append(sum(fitness_scores)/len(fitness_scores))

            # Take worst fitness from each generation
            results_worst.append(fitness_scores[-1])

            g += 1

        # Plot the best fitness across generations:
        plt.title("Performance")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")

        plt.plot(np.arange(0, len(results)), results, label="Best Fitness")
        plt.plot(np.arange(0, len(results_avg)), results_avg, label="Average Fitness")
        #plt.plot(np.arange(0, len(results_worst)), results_worst, label="Worst Fitness")
        plt.legend()

        # Fit the trend line
        """z = np.polyfit(np.arange(0, len(results)), results, 3)
        p = np.poly1d(z)
        plt.plot(np.arange(0, len(results)), p(np.arange(0, len(results))))"""

        #plt.ylim(0,10.5)
        plt.show()
        exit()

        gene_count = 0
        regulatory_matrices = None
        g = 0
        controller_path = os.path.join('experiment_data', exp_name, f'generation_{g}', 'controller', f'1.npz')
        while os.path.exists(controller_path):
            # Load controller data
            controller_data = np.load(controller_path)
            controller = []
            for key, value in controller_data.items():
                controller.append(value)
            controller = tuple(controller)
            gene_count, interaction_matrix = controller

            if regulatory_matrices is None:
                regulatory_matrices = []
                for n in range(gene_count * gene_count):
                    regulatory_matrices.append([])

            for n in range(gene_count * gene_count):
                regulatory_matrices[n].append(interaction_matrix[n])

            """save_path = 'images'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save images of interaction matrix over generations
            matrix_2d = np.zeros((gene_count, gene_count))
            for i in range(gene_count):
                for j in range(gene_count):
                    matrix_2d[i][j] = interaction_matrix[i * gene_count + j]
            cmap = LinearSegmentedColormap.from_list('', ['#000', '#FFF'])
            plt.matshow(matrix_2d, cmap=cmap, vmin=matrix_2d.min(), vmax=matrix_2d.max())
            plt.colorbar()
            plt.savefig(f'images/{g}.png')"""

            g += 1
            controller_path = os.path.join('experiment_data', exp_name, f'generation_{g}', 'controller', '1.npz')


        # Plot regulatory matrix graph
        plt.title("Line graph")
        plt.xlabel("Generations")
        plt.ylabel("Regulatory interaction coefficient")

        for n in range(gene_count * gene_count):
            plt.plot(np.arange(1, g + 1), regulatory_matrices[n], label="Regulation", color=np.random.rand(3, ))

        # plt.ylim(0)
        # plt.legend()
        plt.show()

