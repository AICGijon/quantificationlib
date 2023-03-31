"""
Classes and functions for generating bags of examples and distributions of bags with different kind of drifts
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 clause, University of Oviedo

import numpy as np
import numbers

from abc import ABCMeta

from sklearn.utils import check_X_y
from sklearn.metrics.pairwise import rbf_kernel

from quantificationlib.multiclass.df import compute_bincuts


class BagGenerator(metaclass=ABCMeta):
    """ Base class for bag generator classes
    """
    def generate_bags(self, X, y):
        raise NotImplementedError()


class PriorShift_BagGenerator(BagGenerator):
    """ Generate bags with prior probability shift

        Parameters
        ----------
        n_bags : int, (default=1000)
            Number of bags

        bag_size : int, (default=None)
            Number of examples in each bag

        method : str, (default='Uniform')
            Method used to generate the distributions. Two methods available:
            - 'Uniform' : the prevalences are uniformly distributed
            - 'Dirichlet : the prevalences are generated using the Dirichlet distribution

        alphas : None, float or array-like, (default=None), shape (n_classes, ) when it is an array
            The parameters for the Dirichlet distribution when the selected method is 'Dirichlet'

        min_prevalence : None, float or array-like, (default=None)
            The min prevalence for each class. If None the min prevalence will be 0. If just a single value is passed
            all classes have the same min_prevalence value. This parameter is only used when 'Uniform' method
            is selected

        random_state : int, RandomState instance, (default=2032)
            To generate random numbers
            If type(random_state) is int, random_state is the seed used by the random number generator;
            If random_state is a RandomState instance, random_state is the own random number generator;

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        n_bags : int
            Number of bags

        bag_size : int
            Number of examples in each bag

        method : str
            Method used to generate the prevalences

        alphas : None, float or array-like
            Parameters of the Dirichlet distribution

        min_prevalence :  None, float or array-like
            The min prevalence for each class

        random_state : int, RandomState instance
            To generate random numbers

        verbose : int, optional
            The verbosity level

        prevalences_ : array-like, shape (n_classes, n_bags)
            i-th row contains the true prevalences of each generated bag

        indexes_ : array-line, shape (bag_size, n_bags)
            i-th column contains the indexes of the examples for i-th bag
    """

    def __init__(self, n_bags=1000, bag_size=None, method='Uniform',
                 alphas=None, min_prevalence=None, random_state=2032, verbose=0):
        # attributes
        self.n_bags = n_bags
        self.bag_size = bag_size
        self.method = method
        self.alphas = alphas
        self.min_prevalence = min_prevalence
        self.random_state = random_state
        self.verbose = verbose
        #  variables to represent the bags
        self.prevalences_ = None
        self.indexes_ = None

    def generate_bags(self, X, y):
        """ Create bags of examples simulating prior probability shift

            Two different methods are implemented:
            - 'Uniform' : the prevalences are uniformly distributed
            - 'Dirichlet : the prevalences are generated using the Dirichlet distribution

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            Returns
            ------
            prevalences : numpy array, shape (n_bags, n_classes)
                Each row contains the prevalences of the corresponding bag

            indexes : numpy array, shape (size_bags, n_bags)
                Each column contains the indexes of the examples of the bag

            Raises
            ------
            ValueError
                When random_state is neither a int nor a RandomState object, when the selected method
                is not implemented or when the parameters for the selected method are incorrect
        """

        if self.prevalences_ is not None and self.indexes_ is not None:
            return self.prevalences_, self.indexes_

        if isinstance(self.random_state, (numbers.Integral, np.integer)):
            self.random_state = np.random.RandomState(self.random_state)
        if not isinstance(self.random_state, np.random.RandomState):
            raise ValueError('Invalid random generaror object')

        if self.bag_size is None:
            self.bag_size = len(y)

        if self.method == 'Uniform':
            return self._generate_uniform(X, y)
        elif self.method == 'Dirichlet':
            return self._generate_dirichlet(X, y)
        else:
            raise ValueError('Invalid method for random generaror object')

    def _generate_uniform(self, X, y):
        """ Create bags of examples simulating prior probability shift using an uniform distribution of the prevalences

            The implemented algorithm was proposed by Kramer, see references

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            Returns
            ------
            prevalences : numpy array, shape (n_bags, n_classes)
                Each row contains the prevalences of the corresponding bag

            indexes : numpy array, shape (size_bags, n_bags)
                Each column contains the indexes of the examples of the bag

            References
            ----------
            http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf

            http://blog.geomblog.org/2005/10/sampling-from-simplex.html
        """

        # checking if min_prevalence has correct values
        if isinstance(self.min_prevalence, list):
            self.min_prevalence = np.array(self.min_prevalence)
        if self.min_prevalence is not None:
            if np.isscalar(self.min_prevalence):
                if self.min_prevalence > 1 or self.min_prevalence < 0:
                    raise ValueError('The value for min_prevalence must be in [0, 1]')
            else:
                if np.sum(self.min_prevalence) > 1:
                    raise ValueError('The values for min_prevalence cannot sum more than 1')
                if (self.min_prevalence < 0).any():
                    raise ValueError('The values for min_prevalence must be greater than 0')

        if X is not None:
            X, y = check_X_y(X, y)

        classes = np.unique(y)
        n_classes = len(classes)

        #  computing min number of examples for each class and low value (for the first class) and the
        #  high value (for the last class)
        #  The idea is to perform a efficient comprobation. For binary problems all random generation will be correct
        #  becuase the values low-high are fixed appropriately. For multiclass problems (min_examples will be
        #  not None) we need to check it
        if self.min_prevalence is None:
            min_nexamples = None
            low = 0
            high = self.bag_size
        elif n_classes == 2:
            #  binary problem
            if np.isscalar(self.min_prevalence):
                # same prevalence
                min_nexamples = None
                low = round(self.bag_size * self.min_prevalence)
                high = int(self.bag_size * (1 - self.min_prevalence))
            else:
                # different prevalence
                min_nexamples = None
                low = round(self.bag_size * self.min_prevalence[0])
                high = int(self.bag_size * (1 - self.min_prevalence[1]))
        elif np.isscalar(self.min_prevalence):
            #  multiclass problem, min prevalence equal to all classes
            min_nexamples = np.round(np.ones(n_classes) * self.min_prevalence * self.bag_size)
            low = min_nexamples[0]
            high = self.bag_size - min_nexamples[-1]
        else:
            #  multiclass problem, different prevalences
            min_nexamples = np.round(self.min_prevalence * self.bag_size)
            low = min_nexamples[0]
            high = self.bag_size - min_nexamples[-1]

        self.prevalences_ = np.zeros((n_classes, self.n_bags))
        self.indexes_ = np.zeros((self.bag_size, self.n_bags), dtype=np.uint32)
        n_bag = 0
        while n_bag < self.n_bags:
            # Kraemer method:
            ps = self.random_state.randint(low, high, n_classes - 1)
            ps = np.append(ps, [0, self.bag_size])
            ps = np.diff(np.sort(ps))  # number of samples for each class

            #  checking if the combination is correct, only needed in multiclass problems. Only the values for the
            #  classes in the middle must be checked
            if min_nexamples is None or (ps[1:-1] >= min_nexamples[1:-1]).all():
                self.prevalences_[:, n_bag] = ps / self.bag_size  # to obtain prevalences
                position = 0
                for cls, n_examples in zip(classes, np.array(ps)):
                    if n_examples != 0:
                        indexes = self.random_state.choice(np.where(y == cls)[0], n_examples, replace=True)
                        self.indexes_[position:position+len(indexes), n_bag] = indexes
                        position = position + len(indexes)

                n_bag = n_bag + 1
        return self.prevalences_, self.indexes_

    def _generate_dirichlet(self, X, y):
        """ Create bags of examples simulating prior probability shift using the Dirichlet distribution

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            Returns
            ------
            prevalences : numpy array, shape (n_bags, n_classes)
                Each row contains the prevalences of the corresponding bag

            indexes : numpy array, shape (size_bags, n_bags)
                Each column contains the indexes of the examples of the bag
        """
        if X is not None:
            X, y = check_X_y(X, y)

        classes = np.unique(y)
        n_classes = len(classes)

        # checking if alphas  has correct values
        if isinstance(self.alphas, list):
            self.alphas = np.array(self.alphas)
        if self.alphas is not None:
            if np.isscalar(self.alphas):
                self.alphas = np.ones(n_classes) * self.alphas
            else:
                if len(self.alphas) != n_classes:
                    raise ValueError('the size of alphas parameter does not match the number of classes')

        self.prevalences_ = np.random.dirichlet(self.alphas, size=self.n_bags).transpose()
        self.indexes_ = np.zeros((self.bag_size, self.n_bags), dtype=np.uint32)

        for n_bag in range(self.n_bags):
            position = 0
            for n_cls, cls in enumerate(classes):
                #  compute the number of examples given the prevalence
                if n_cls < n_classes - 1:
                    n_examples = min(self.bag_size - position,
                                     int(np.round(self.prevalences_[n_cls, n_bag] * self.bag_size)))
                else:
                    n_examples = self.bag_size - position
                #  adjust the true prevalence given the number of examples
                self.prevalences_[n_cls, n_bag] = n_examples / self.bag_size

                if n_examples != 0:
                    indexes = self.random_state.choice(np.where(y == cls)[0], n_examples, replace=True)
                    self.indexes_[position:position + len(indexes), n_bag] = indexes
                    position = position + len(indexes)

        return self.prevalences_, self.indexes_


class CovariateShift_BagGenerator(BagGenerator):
    """ Generate bags with covariate shift

        The idea is to pick just an instance from X and then randomly selecting the examples of the bag according
        to their distance to said instance

        Parameters
        ----------
        n_bags : int, (default=1000)
            Number of bags

        bag_size : int, (default=None)
            Number of examples in each bag

        random_state : int, RandomState instance, (default=2032)
            To generate random numbers
            If type(random_state) is int, random_state is the seed used by the random number generator;
            If random_state is a RandomState instance, random_state is the own random number generator;

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        n_bags : int
            Number of bags

        bag_size : int
            Number of examples in each bag

        random_state : int, RandomState instance
            To generate random numbers

        verbose : int, optional
            The verbosity level

        prevalences_ : array-like, shape (n_classes, n_bags)
            i-th row contains the true prevalences of each generated bag

        indexes_ : array-line, shape (bag_size, n_bags)
            i-th column contains the indexes of the examples for i-th bag
    """

    def __init__(self, n_bags=1001, bag_size=None, random_state=2032, verbose=0):

        # attributes
        self.n_bags = n_bags
        self.bag_size = bag_size
        self.random_state = random_state
        self.verbose = verbose
        #  variables to represent the bags
        self.prevalences_ = None
        self.indexes_ = None

    def generate_bags(self, X, y):
        """ Create bags of examples simulating covariate shift

            The method first picks a center example for each bag. The probability to select an example for the bag
            is proportional to the distance to the centrr example.

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            Returns
            ------
            prevalences : numpy array, shape (n_bags, n_classes)
                Each row contains the prevalences of the corresponding bag

            indexes : numpy array, shape (size_bags, n_bags)
                Each column contains the indexes of the examples of the bag

            Raises
            ------
            ValueError
                When random_state is neither a int nor a RandomState object
        """

        if self.prevalences_ is not None and self.indexes_ is not None:
            return self.prevalences_, self.indexes_

        if isinstance(self.random_state, (numbers.Integral, np.integer)):
            self.random_state = np.random.RandomState(self.random_state)
        if not isinstance(self.random_state, np.random.RandomState):
            raise ValueError('Invalid random generaror object')

        X, y = check_X_y(X, y)
        classes = np.unique(y)
        n_classes = len(classes)
        if self.bag_size is None:
            self.bag_size = len(X)

        self.prevalences_ = np.zeros((n_classes, self.n_bags))
        self.indexes_ = np.zeros((self.bag_size, self.n_bags), dtype=np.uint32)

        #  selecting the center example for each bag
        centers = self.random_state.randint(0, len(X), self.n_bags)

        for n_bag in range(self.n_bags):
            #  compute similarity using RBF kernel using the default value for gamma 1/num_features
            similarity = rbf_kernel(X[centers[n_bag], :].reshape(1, -1), X)
            #  the probability to select an example for the bag is proportional to its similarity
            probs = similarity / np.sum(similarity)

            self.indexes_[:, n_bag] = self.random_state.choice(range(len(X)), self.bag_size, p=probs[0, :])
            for n_cls, cls in enumerate(classes):
                self.prevalences_[n_cls, n_bag] = np.sum(y[self.indexes_[:, n_bag]] == cls) / self.bag_size

        return self.prevalences_, self.indexes_


class PriorAndCovariateShift_BagGenerator(BagGenerator):
    """ Generate bags with a mix of prior probability shift and covariate shfit

        This class generates the bags using two objects of the classes PriorShift_BagGenerator and
        CovariateShift_BagGenerator

        Parameters
        ----------
        n_bags : int, (default=1000)
            Number of bags

        bag_size : int, (default=None)
            Number of examples in each bag

        method : str, (default='Uniform')
            Method used to generate the distributions. Two methods available:
            - 'Uniform' : the prevalences are uniformly distributed
            - 'Dirichlet : the prevalences are generated using the Dirichlet distribution

        alphas : None, float or array-like, (default=None), shape (n_classes, ) when it is an array
            The parameters for the Dirichlet distribution when the selected method is 'Dirichlet'

        min_prevalence : None, float or array-like, (default=None)
            The min prevalence for each class. If None the min prevalence will be 0. If just a single value is passed
            all classes have the same min_prevalence value. This parameter is only used when 'Uniform' method
            is selected

        random_state : int, RandomState instance, (default=2032)
            To generate random numbers
            If type(random_state) is int, random_state is the seed used by the random number generator;
            If random_state is a RandomState instance, random_state is the own random number generator;

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        n_bags : int
            Number of bags

        bag_size : int
            Number of examples in each bag

        min_prevalence:  None, float or array-like
            The min prevalence for each class.

        random_state : int, RandomState instance
            To generate random numbers

        verbose : int, optional
            The verbosity level

        prevalences_ : array-like, shape (n_classes, n_bags)
            i-th row contains the true prevalences of each generated bag

        indexes_ : array-line, shape (bag_size, n_bags)
            i-th column contains the indexes of the examples for i-th bag
    """

    def __init__(self, n_bags=1000, bag_size=None, method='Uniform', alphas=None,
                 min_prevalence=None, random_state=2032, verbose=0):

        # attributes
        self.n_bags = n_bags
        self.bag_size = bag_size
        self.method = method
        self.alphas = alphas
        self.min_prevalence = min_prevalence
        self.random_state = random_state
        self.verbose = verbose
        #  variables to represent the training bags and quantifiers
        self.prevalences_ = None
        self.indexes_ = None

    def generate_bags(self, X, y):
        """ Create bags of examples simulating prior probability shift and covariate shift. It uses instances of
            classes PriorShift_BagGenerator and CovariateShift_BagGenerator

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            Returns
            ------
            prevalences : numpy array, shape (n_bags, n_classes)
                Each row contains the prevalences of the corresponding bag

            indexes : numpy array, shape (size_bags, n_bags)
                Each column contains the indexes of the examples of the bag

            Raises
            ------
            ValueError
                When random_state is neither a int nor a RandomState object
        """

        if self.prevalences_ is not None and self.indexes_ is not None:
            return self.prevalences_, self.indexes_

        n_bags_prior = self.n_bags // 2
        n_bags_covariate = self.n_bags - n_bags_prior
        prior = PriorShift_BagGenerator(n_bags=n_bags_prior, bag_size=self.bag_size,
                                        method=self.method, alphas=self.alphas,
                                        min_prevalence=self.min_prevalence, random_state=self.random_state,
                                        verbose=self.verbose)
        covariate = CovariateShift_BagGenerator(n_bags=n_bags_covariate, bag_size=self.bag_size,
                                                random_state=self.random_state,
                                                verbose=self.verbose)

        classes = np.unique(y)
        n_classes = len(classes)
        self.prevalences_ = np.zeros((n_classes, self.n_bags))
        self.indexes_ = np.zeros((self.bag_size, self.n_bags), dtype=np.uint32)

        self.prevalences_[:, 0:n_bags_prior], self.indexes_[:, 0:n_bags_prior] = prior.generate_bags(X, y)
        self.prevalences_[:, n_bags_prior:], self.indexes_[:, n_bags_prior:] = covariate.generate_bags(X, y)

        return self.prevalences_, self.indexes_
