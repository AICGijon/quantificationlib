import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

# sort rows based on the list of columns
# by default sort based on the first column ascending
def sort_rows_bycolumns(m, columns=[0], order='ascending'):
    columns.reverse()  #jojo!!
    sort_order = np.lexsort(tuple(m[:, i] for i in columns))
    if order=='descending':
        sort_order = sort_order[::-1]
    m_sorted = m[sort_order]
    return m_sorted

# sort columns based on the means of its values
# return the new matrix and the index of the columns
def sort_columns_bymeans(m, order='ascending'):
    means = np.mean(m, axis=0)
    print("means:",means)
    idx_means = np.argsort(means)
    if order=='descending':
        idx_means = idx_means[::-1]
    print(idx_means)
    m_new = m[:, idx_means]  #consider inline ordering??
    return m_new, idx_means


def plot_line_prevalences(ax, prevalence_matrix, order=None, colors=list(mcolors.TABLEAU_COLORS), param_dict=''):
    """
    Function to build a graph to represent prevalences of experiments in a linear form.
    Each horizontal line corresponds to one experiment, each color is a class
    and its horizontal length corresponds to its proportion (prevalence)

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    prevalence_matrix : ndarray, shape(number_of_experiments, number_ of_clases)
        Prevalences of the experiments
        Example for 4 classes:
        [[0.01974937 0.87850524 0.0584321  0.04331329]
        ...
        [0.47830422 0.09838137 0.10653123 0.31678318]]

    order: str    
        Sort matrix based on the first column in order "ascending", "desdecending" or none
       
    colors: list
        List of colors for classes

    param_dict : dict
        Dictionary of keyword arguments to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added  ???????
    """

    if order!=None:
       prevalence_matrix = sort_rows_bycolumns(prevalence_matrix, order=order)


    N = 100  # points per line
    n_rows = prevalence_matrix.shape[0] 
    ax.set_xlim([0, N])
    ax.set_ylim([0, (n_rows+1)])
    ax.set(xticks = [], yticks = [])

    for i in range(0,n_rows): # each row is a line in the graph
           segments=[]        # each line is a list of colored segments
           y=0+(i+1)
           curr_prevs=prevalence_matrix[i]   # Ex: [0.01974937 0.87850524 0.0584321  0.04331329]
           x0=0.0
           for prev in curr_prevs:
               num_pts=N*prev
               x1=x0+num_pts
               segments.append([(x0,y),(x1,y)])
               x0=x1
           #print(segments)
           line_segments = LineCollection(segments, colors=colors, linewidths=2)
           ax.add_collection(line_segments)

    return ax