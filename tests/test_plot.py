from quantificationlib.plot.qlplot import plot_line_prevalences, plot_boxes
import matplotlib.pyplot as plt 
from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.data_utils import load_data
import numpy as np
import matplotlib.colors as mcolors

def test_plots():
    dataset='examples/datasets/binary/iris.3.csv'
        
    X, y = load_data(dataset)

    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    bag_generator = PriorShift_BagGenerator(n_bags=50, bag_size=100, min_prevalence=None, random_state=2032)

    prev_true, _ = bag_generator.generate_bags(X, y)

    plot_line_prevalences(ax, prevalence_matrix=prev_true, order='descending')

    results = [np.random.rand(50), np.random.rand(50), np.random.rand(50)]
    methods = ["Method 1", "Method 2", "Method 3"]

    fig2, axs2 = plt.subplots(ncols=2, figsize=(10,4))
    plot_boxes(axs2[0], results, vert=None, y_title='MAE', x_title=None,
                labels=methods, colors=None) 
    plot_boxes(axs2[1], results, vert=False, y_title=None, x_title='MAE',
                labels=methods, colors=list(mcolors.TABLEAU_COLORS))


