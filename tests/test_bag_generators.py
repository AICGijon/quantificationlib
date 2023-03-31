import numpy as np

from quantificationlib.bag_generator import PriorShift_BagGenerator, CovariateShift_BagGenerator, PriorAndCovariateShift_BagGenerator
from quantificationlib.data_utils import load_data

def test_bag_generator_uniform():
    master_seed = 2032
    dataset='examples/datasets/binary/iris.3.csv'

    X, y = load_data(dataset)

    bag_generator = PriorShift_BagGenerator(n_bags=1000, bag_size=100,
                                            min_prevalence=None, random_state=master_seed)
    bags = bag_generator.generate_bags(X, y)
    assert bags[0].shape==(2, 1000), "Error in prior bag generator"
    assert bags[1].shape==(100, 1000), "Error in prior bag generator"

    bag_generator = PriorShift_BagGenerator(n_bags=1000, bag_size=100, method='Dirichlet',alphas=(0.1,0.1),
                                            min_prevalence=None, random_state=master_seed)
    bags = bag_generator.generate_bags(X, y)
    assert bags[0].shape==(2, 1000), "Error in prior bag generator"
    assert bags[1].shape==(100, 1000), "Error in prior bag generator"

def test_bag_generator_dirichlet():
    master_seed = 2032
    dataset='examples/datasets/binary/iris.3.csv'

    X, y = load_data(dataset)

    bag_generator = PriorShift_BagGenerator(n_bags=1000, bag_size=100, method='Dirichlet',alphas=(0.1,0.1),
                                            min_prevalence=None, random_state=master_seed)
    bags = bag_generator.generate_bags(X, y)
    assert bags[0].shape==(2, 1000), "Error in prior bag generator"
    assert bags[1].shape==(100, 1000), "Error in prior bag generator"

def test_min_prevalence():
    master_seed = 2032
    dataset='examples/datasets/binary/iris.3.csv'

    X, y = load_data(dataset)

    bag_generator = PriorShift_BagGenerator(n_bags=1000, bag_size=100,
                                            min_prevalence=0.1, random_state=master_seed)
    
    
    bags = bag_generator.generate_bags(X, y)
    assert bags[0].shape==(2, 1000), "Error in prior bag generator"
    assert bags[1].shape==(100, 1000), "Error in prior bag generator"
    assert np.all(bags[0] >= 0.1)

def test_covariate_shift_bag_generator():
    master_seed = 2032
    dataset='examples/datasets/binary/iris.3.csv'

    X, y = load_data(dataset)

    bag_generator = CovariateShift_BagGenerator(n_bags=1000, bag_size=100, random_state=master_seed)
    
    bags = bag_generator.generate_bags(X, y)
    assert bags[0].shape==(2, 1000), "Error in covariate bag generator"
    assert bags[1].shape==(100, 1000), "Error in covariate bag generator"

def test_covariate_and_prior_shift_bag_generator():
    master_seed = 2032
    dataset='examples/datasets/binary/iris.3.csv'

    X, y = load_data(dataset)

    bag_generator = PriorAndCovariateShift_BagGenerator(n_bags=1000, bag_size=100, random_state=master_seed)
    
    bags = bag_generator.generate_bags(X, y)
    assert bags[0].shape==(2, 1000), "Error in prior covariate bag generator"
    assert bags[1].shape==(100, 1000), "Error in prior covariate bag generator"
