import random
import numpy as np
from robot import Robot
from ga import GeneticAlgorithm
from grn import WatsonGRN, CrossoverGRN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

if __name__ == '__main__':
    gene_count = 5
    
    parent_one = WatsonGRN(gene_count)
    parent_one.interaction_matrix = np.zeros(gene_count * gene_count)
    for i in range(gene_count * gene_count):
        if np.floor(i / gene_count) % 2 == 0:
            parent_one.interaction_matrix[i] = 1
        else:
            parent_one.interaction_matrix[i] = -1
            
    parent_two = WatsonGRN(gene_count)
    parent_two.interaction_matrix = np.zeros(gene_count * gene_count)
    for i in range(gene_count * gene_count):
        if (i * gene_count) % 2 == 0:
            parent_two.interaction_matrix[i] = 1
        else:
            parent_two.interaction_matrix[i] = -1
            
            
    child = CrossoverGRN(parent_one, parent_two)
            
    
    # Plot matrix of evolved regulatory interactions
    parent_one_plot = np.zeros((gene_count, gene_count))

    for i in range(gene_count):
        for j in range(gene_count):
            parent_one_plot[i][j] = parent_one.interaction_matrix[i * gene_count + j]

    cmap = LinearSegmentedColormap.from_list('', ['#000', '#FFF'])
    plt.matshow(parent_one_plot, cmap=cmap, vmin=parent_one_plot.min(), vmax=parent_one_plot.max())
    plt.colorbar()
    plt.show()
    
    # Plot matrix of evolved regulatory interactions
    parent_two_plot = np.zeros((gene_count, gene_count))

    for i in range(gene_count):
        for j in range(gene_count):
            parent_two_plot[i][j] = parent_two.interaction_matrix[i * gene_count + j]

    cmap = LinearSegmentedColormap.from_list('', ['#000', '#FFF'])
    plt.matshow(parent_two_plot, cmap=cmap, vmin=parent_two_plot.min(), vmax=parent_two_plot.max())
    plt.colorbar()
    plt.show()
    
    # Plot matrix of evolved regulatory interactions
    child_plot = np.zeros((gene_count, gene_count))

    for i in range(gene_count):
        for j in range(gene_count):
            child_plot[i][j] = child.interaction_matrix[i * gene_count + j]

    cmap = LinearSegmentedColormap.from_list('', ['#000', '#FFF'])
    plt.matshow(child_plot, cmap=cmap, vmin=child_plot.min(), vmax=child_plot.max())
    plt.colorbar()
    plt.show()