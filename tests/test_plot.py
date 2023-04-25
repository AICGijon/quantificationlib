from quantificationlib.plot.qlplot import plot_line_prevalences
import matplotlib.pyplot as plt 
from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.data_utils import load_data

dataset='examples/datasets/binary/iris.3.csv'
    
X, y = load_data(dataset)

fig = plt.figure(figsize=(15, 4))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

bag_generator = PriorShift_BagGenerator(n_bags=50, bag_size=100, min_prevalence=None, random_state=2032)

prev_true, _ = bag_generator.generate_bags(X, y)

plot_line_prevalences(ax, prevalence_matrix=prev_true)


