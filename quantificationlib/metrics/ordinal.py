"""
Score functions and loss functions for ordinal quantification problems
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from sklearn.metrics.pairwise import check_pairwise_arrays

from quantificationlib.metrics.multiclass import check_prevalences


def emd(p_true, p_pred):
    """ Return the EMD distances between two sets of prevalences

        Parameters
        ----------
        p_true: array-like, shape (n_classes, 1)

        p_pred: array-like, shape (n_classes, 1)

        Return
        ------
        emd : float, the EMD distance between p_true and p_pred
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sum(np.abs(np.cumsum(p_true - p_pred)))
    # return sum([abs(sum(p_pred[:j+1]) - sum(p_true[:j+1])) for j in range(len(p_true)-1)])


def emd_score(p_true, p_pred):
    p_true, p_pred = check_prevalences(p_true, p_pred)
    left_mass = np.zeros_like(p_true)
    left_mass[0] = 1
    right_mass = np.zeros_like(p_true)
    right_mass[-1] = 1
    max_emd = max(emd(p_true, left_mass), emd(p_true, right_mass))
    return (max_emd - np.sum(np.abs(np.cumsum(p_true - p_pred)))) / max_emd


def emd_distances(A, B):
    """ Return the EMD distances between the rows of A and B

        Parameters
        ----------
        A: array-like, shape (n_samples_1, n_features)

        B: array-like, shape (n_samples_2, n_features)

        A and B should have the same number of columns

        Return
        ------
        distances : array, shape (n_samples_1, n_samples_2)
    """
    # version #1: the most inefficient
    # return np.sum(np.abs(np.cumsum(A[:, np.newaxis] - B, axis=2)), axis=2)

    # version #2: this is more efficient that version#1 but a little bit worse than version#3
    # differences = A[:, 0].reshape(-1, 1) - B[:, 0].reshape(-1, 1).T
    # distances = np.abs(differences)
    # for i in range(1, A.shape[1]):
    #     differences = A[:, i].reshape(-1, 1) - B[:, i].reshape(-1, 1).T + differences
    #     distances = distances + np.abs(differences)
    # return distances

    # version #3: the most efficient one
    A, B = check_pairwise_arrays(A, B)
    A = np.cumsum(A, axis=1)
    B = np.cumsum(B, axis=1)
    distances = np.abs(A[:, 0].reshape(-1, 1) - B[:, 0].reshape(-1, 1).T)
    for i in range(1, A.shape[1]-1):
        distances = distances + np.abs(A[:, i].reshape(-1, 1) - B[:, i].reshape(-1, 1).T)
    return distances


