"""
Search functions. Needed by those quantifiers based on matching distribution that compute the estimated prevalence
using a search algorithm
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np
import math
import warnings

from scipy.stats import norm


############
# Search function for distance functions that can have several local minima
############
def global_search(distance_func, mixture_func, test_distrib, tol, mixtures, return_mixtures, **kwargs):
    """ Search function for non-V-shape distance functions

        Given a function `distance_func` with a single local minumum in the interval [0,1], the method
        returns the prevalence that minimizes the differente between the mixture training distribution and
        the testing distribution according to `distance_func`

        This method is based on using Golden Section Search but this kind of search only works when loss is V shape.
        We found that same combinations of quantifiers/loss functions do no produce a V shape. Instead of just checking
        that, this method first computes the loss for all the points in the range [0, 1] with a step of 0.01. Then,
        around all the minimums a Golden Section Search is performed to find the global minimum

        Used by QUANTy, SORDy and DF-based classes. Only useful for binary quantification

        Parameters
        ----------
        distance_func : function
            This is the loss function minimized during the search

        mixture_func : function
            The function used to generated the training mixture distribution given a value for the prevalence

        test_distrib : array
            The distribution of the positive class. The exact shape depends on the representation (pdfs, quantiles...)

        tol : float
            The precision of the solution

        mixtures : array
            Contains the mixtures for all the prevalences in the range [0, 1] step 0.01. This mixtures can be computed
            just once, for the first testing bag, and applied for the rest. It is useful when computing the mixture is
            time consuming. Only used by QUANTy.

        return_mixtures : boolean
            Contains True if the method must return the precomputed_mixtures

        kwargs : keyword arguments
            Here we pass the set of arguments needed by mixture functions: mixture_two_pdfs (for pdf-based classes) and
            compute quantiles (for quantiles-based classes). See the help of this two functions

        Returns
        -------
        mixtures: array or None
            Computed mixtures for the range [0, 1] step 0.01

        prevalences : array, shape(2,)
            The predicted prevalence for the negative and the positive class
    """
    losses = np.ones(101)

    if mixtures is None and return_mixtures:
        mix_0 = mixture_func(prevalence=0, **kwargs)
        mixtures = np.zeros((101, len(mix_0)))
        mixtures[0, :] = np.squeeze(mix_0)
        losses[0] = distance_func(mixtures[0, :], test_distrib)
        for i in range(1, 101):
            mixtures[i, :] = np.squeeze(mixture_func(prevalence=i / 100, **kwargs))
            losses[i] = distance_func(mixtures[i, :], test_distrib)
    elif mixtures is not None:
        for i in range(0, 101):
            losses[i] = distance_func(mixtures[i, :], test_distrib)
    else:
        for i in range(0, 101):
            losses[i] = distance_func(mixture_func(prevalence=i / 100, **kwargs), test_distrib)

    n_mins = 0
    p_estimated = 2
    minimum_loss = np.inf
    for i in range(1, 100):
        if losses[i - 1] < losses[i] < losses[i + 1]:
            n_mins = n_mins + 1
            loss, p = golden_section_search(distance_func, mixture_func, test_distrib, tol,
                                            max(i - 2, 0) / 100, min(i + 2, 100) / 100, **kwargs)
            if loss < minimum_loss:
                minimum_loss = loss
                p_estimated = p

    if n_mins == 0:
        if losses[0] == losses[100]:
            raise ValueError("Search: there are infinite minima according to the distance function used")
        elif losses[0] < losses[100]:
            _, p_estimated = golden_section_search(distance_func, mixture_func, test_distrib, tol, 0.0, 0.02, **kwargs)
        else:
            _, p_estimated = golden_section_search(distance_func, mixture_func, test_distrib, tol, 0.98, 1.0, **kwargs)

    if return_mixtures:
        return mixtures, p_estimated
    else:
        return None, p_estimated


############
# Search function for distance functions with just one global minimum
############
def golden_section_search(distance_func, mixture_func, test_distrib, tol, a, b, **kwargs):
    """ Golden section search

        Only useful for binary quantification
        Given a function `distance_func` with a single local minumum in the interval [a, b], `golden_section_search`
        returns the prevalence that minimizes the differente between the mixture training distribution and
        the testing distribution according to `distance_func`

        Parameters
        ----------
        distance_func : function
            This is the loss function minimized during the search

        mixture_func : function
            The function used to generated the training mixture distribution given a value for the prevalence

        test_distrib : array
            The distribution of the positive class. The exact shape depends on the representation (pdfs, quantiles...)

        tol : float
            The precision of the solution

        a : float
            The lower bound of the interval

        b: float
            The upper bound of the interval

        kwargs : keyword arguments
            Here we pass the set of arguments needed by mixture functions: mixture_two_pdfs (for pdf-based classes) and
            compute quantiles (for quantiles-based classes). See the help of this two functions

        Returns
        -------
        loss : float
            Distance between mixture and testing distribution for the returned prevalence according to distance_func

        prevalences : array, shape(2,)
            The predicted prevalence for the negative and the positive class
    """

    # some constants
    invphi = (math.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1/phi^2

    h = b - a

    # required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h

    train_mixture_distrib = mixture_func(prevalence=c, **kwargs)
    fc = distance_func(train_mixture_distrib, test_distrib)
    train_mixture_distrib = mixture_func(prevalence=d, **kwargs)
    fd = distance_func(train_mixture_distrib, test_distrib)

    for k in range(n - 1):
        if fc < fd:
            b = d
            d = c
            fd = fc
            h = invphi * h
            c = a + invphi2 * h
            train_mixture_distrib = mixture_func(prevalence=c, **kwargs)
            try:
                fc = distance_func(train_mixture_distrib, test_distrib)
            except:
                print(train_mixture_distrib)
                print(c)
                for key, value in kwargs.items():
                    print("%s == %s" % (key, value))
        else:
            a = c
            c = d
            fc = fd
            h = invphi * h
            d = a + invphi * h
            train_mixture_distrib = mixture_func(prevalence=d, **kwargs)
            try:
                fd = distance_func(train_mixture_distrib, test_distrib)
            except:
                print(train_mixture_distrib)
                print(c)
                for key, value in kwargs.items():
                    print("%s == %s" % (key, value))

    if fc < fd:
        return fc, np.array([1 - (a + d) / 2, (a + d) / 2])
    else:
        return fd, np.array([1 - (c + b) / 2, (c + b) / 2])


############
# Functions for computing mixtures of distributions (pdfs and quantiles)
############
def mixture_of_pdfs(prevalence=None, pos_distrib=None, neg_distrib=None):
    """ Mix two pdfs given a value for the prevalence of the positive class

        Parameters
        ----------
        prevalence : float,
           The prevalence for the positive class

        pos_distrib : array, shape(n_bins,)
            The distribution of the positive class. The exact shape depends on the representation (pdfs, quantiles...)

        neg_distrib : array, shape(n_bins,)
            The distribution of the negative class. The exact shape depends on the representation (pdfs, quantiles...)

        Returns
        -------
        mixture : array, same shape of positives and negatives
           The pdf mixture of positives and negatives
    """
    mixture = pos_distrib * prevalence + neg_distrib * (1 - prevalence)
    return mixture


def compute_quantiles(prevalence=None, probabilities=None, n_quantiles=None, y=None, classes=None):
    """ Compute quantiles

        Used by QUANTy. It computes the quantiles both for the testing distribution (in this case
        the value of the prevalence is ignored), and for the weighted mixture of positives and negatives (this depends
        on the value of the prevalence parameter)

        Parameters
        ----------
        prevalence : float or None
            The value of the prevalence of the positive class to compute the mixture of the positives and the negatives.
            To compute the quantiles of the testing set this parameter must be None

        probabilities : ndarray, shape (nexamples, 1)
            The ordered probabilities for all examples. Notice that in the case of computing the mixture of the
            positives and the negatives, this array contains the probability for all the examples of the training set

        n_quantiles : int
            Number of quantiles. This parameter is used with Quantiles-based algorithms.

        y : array, labels
            This parameter is used with Quantiles-based algorithms. They need the true label of each example

        classes: ndarray, shape (n_classes, )
            Class labels. Used by Quantiles-based algorithms

        Returns
        -------
        quantiles : array, shape(n_quantiles,)
           The value of the quantiles given the probabilities (and the value of the prevalence if we are computing the
           quantiles of the training mixture distribution)
    """

    # by default (test set) the weights are all equal
    p_weight = np.ones(len(probabilities))
    if prevalence is not None:
        # train set
        n = 1 - prevalence
        n_negatives = np.sum(y == classes[0])
        n_positives = np.sum(y == classes[1])
        p_weight[y == classes[0]] = n * len(probabilities) / n_negatives
        p_weight[y == classes[1]] = prevalence * len(probabilities) / n_positives

    cutpoints = np.array(range(1, n_quantiles + 1)) / n_quantiles * len(probabilities)

    quantiles = np.zeros(n_quantiles)
    cumweight = 0
    j = 0
    for i in range(len(probabilities)):
        cumweight = cumweight + p_weight[i]
        if cumweight <= cutpoints[j]:
            quantiles[j] = quantiles[j] + probabilities[i] * p_weight[i]
        else:
            quantiles[j] = quantiles[j] + probabilities[i] * (p_weight[i] - (cumweight - cutpoints[j]))
            withoutassign = cumweight - cutpoints[j]
            #  withoutassign could be greater than the weight corresponding to a quantile (very rare case, maybe in
            #  an extreme imbalanced case). In that situation we need to distribute withoutassign weight between more
            #  than one quantile
            while withoutassign > 0.0001 and j < n_quantiles - 1:
                j = j + 1
                assign = min(withoutassign, cutpoints[j] - cutpoints[j - 1])
                quantiles[j] = quantiles[j] + probabilities[i] * assign
                withoutassign = withoutassign - assign

    quantiles = quantiles / cutpoints[0]
    return quantiles


############
# Functions for solving SORDy
############
def compute_sord_weights(prevalence=None, union_labels=None, classes=None):
    """ Computes the weight for each example, depending on the prevalence, to compute afterwards the SORD distance

        Parameters
        ----------
        prevalence : float,
           The prevalence for the positive class

        union_labels  :  ndarray, shape (n_examples_train+n_examples_test, 1)
            Contains the set/or  the label of each prediction. If the prediction corresponds to
            a training example, the value is the true class of such example. If the example belongs to the testing
            distribution, the value is NaN

        classes : ndarray, shape (n_classes, )
            Class labels

        Returns
        -------
        weights : array, same shape of union_labels
           The weight of each example, that is equal to:

           negative class = (1-prevalence)*1/|D^-|
           positive class = prevalence*1/|D^+|
           testing examples  = - 1 / |T|

        References
        ----------
        André Maletzke, Denis dos Reis, Everton Cherman, and Gustavo Batista: Dys: A framework for mixture models
        in quantification. In AAAI 2019, volume 33, pp. 4552–4560. 2019.
    """
    weights = np.zeros((len(union_labels), 1))
    for n_cls, cls in enumerate(classes):
        if n_cls == 0:
            weights[union_labels == cls] = (1 - prevalence) / np.sum(union_labels == cls)
        else:
            weights[union_labels == cls] = prevalence / np.sum(union_labels == cls)
    weights[union_labels == np.max(union_labels)] = -1.0 / np.sum(union_labels == np.max(union_labels))
    return weights


def sord(weights, union_distrib):
    """ Computes the SORD distance for SORDy algorithm for a given union_distribution and the weights of the
        examples (that depends on the prevalence used to compute the mixture of the training distribution).
        This methods correspond to the implementation of Algorithm 1 in (Maletzke et al. 2019)

            Parameters
            ----------
            weights : array, shape (n_examples_train+n_examples_test, 1)  (same shape of union_labels)
               The weight of each example, that is equal to:

               negative class = (1-prevalence)*1/|D^-|
               positive class = prevalence*1/|D^+|
               testing examples  = - 1 / |T|

            union_labels  :  ndarray, shape (n_examples_train+n_examples_test, 1)
                Contains the set/or  the label of each prediction. If the prediction corresponds to
                a training example, the value is the true class of such example. If the example belongs to the testing
                distribution, the value is NaN

            Returns
            -------
            total_cost : float
                SORD distance

            References
            ----------
            André Maletzke, Denis dos Reis, Everton Cherman, and Gustavo Batista: Dys: A framework for mixture models
            in quantification. In AAAI 2019, volume 33, pp. 4552–4560. 2019.
        """
    total_cost = 0
    cum_weights = weights[0]
    for i in range(1, len(weights)):
        delta = union_distrib[i] - union_distrib[i - 1]
        total_cost = total_cost + np.abs(delta * cum_weights)
        cum_weights = cum_weights + weights[i]
    return total_cost
