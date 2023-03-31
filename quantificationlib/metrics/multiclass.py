"""
Score functions and loss functions for multiclass quantification problems
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np
import scipy

from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels


def check_prevalences(p_true, p_pred):
    """ Check that p_true and p_pred are valid and consistent

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        p_true : array-like of shape = (n_classes, 1)
            The converted and validated p_true array

        p_pred : array-like of shape = (n_classes, 1)
            The converted and validated p_pred array
    """
    check_consistent_length(p_true, p_pred)
    p_true = check_array(p_true, ensure_2d=False)
    p_pred = check_array(p_pred, ensure_2d=False)

    if p_true.ndim == 1:
        p_true = p_true.reshape((-1, 1))

    if p_pred.ndim == 1:
        p_pred = p_pred.reshape((-1, 1))

    if p_true.shape[1] != p_pred.shape[1]:
        raise ValueError("p_true and p_pred have different length")

    return p_true, p_pred


def kld(p_true, p_pred, eps=1e-12):
    """ Kullback - Leiber divergence (KLD)

            :math:`kld = \sum_{j=1}^{j=l} p_j \cdot \log{p_j/\hat{p}_j}`

        being l the number of classes.

        Also known as discrimination information, relative entropy or normalized cross-entropy
        (see [Esuli and Sebastiani 2010; Forman 2008]).
        KLD is a special case of the family of f-divergences

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        eps : float,
            To prevent division by 0 and Inf/NaN quant_results

        Returns
        -------
        KLD : float
            It is equal to :math:`\sum_{j=1}^{j=l} p_j \cdot \log{p_j / \hat{p}_j}`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sum(p_true * np.log2(p_true / (p_pred + eps)))


def mean_absolute_error(p_true, p_pred):
    """ Mean absolute error

            :math:`mae = 1/l \sum_{j=1}^{j=l} | p_j - \hat{p}_j |`

        being l the number of classes

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        MAE : float
            It is equal to :math:`1/l \sum_{j=1}^{j=l} | p_j - \hat{p}_j |`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.mean(np.abs(p_pred - p_true))


def l1(p_true, p_pred):
    """ L1 loss function

            :math:`l1 = \sum_{j=1}^{j=l} | p_j - \hat{p}_j |`

        being l the number of classes

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        l1 : float
            It is equal to :math:`\sum_{j=1}^{j=l} | p_j - \hat{p}_j |`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sum(np.abs(p_true - p_pred))


def mean_squared_error(p_true, p_pred):
    """ Mean squared error

            :math:`mse = 1/l \sum_{j=1}^{j=l} (p_j - \hat{p}_j)^2`

        being l the number of classes

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        MSE : float
            It is equal to :math:`1/l \sum_{j=1}^{j=l} (p_j - \hat{p}_j)^2`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.mean((p_pred - p_true)**2)


def l2(p_true, p_pred):
    """ L2 loss function

            :math:`l2 = \sqrt{\sum_{j=1}^{j=l} (p_j - \hat{p}_j)^2}`

        being l the number of classes

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        l2 : float
            It is equal to :math:`\sqrt{\sum_{j=1}^{j=l} (p_j - \hat{p}_j)^2}`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sqrt(np.sum((p_true - p_pred) ** 2))


def hd(p_true, p_pred):
    """ Hellinger distance (HD)

            :math:`hd = \sqrt{\sum_{j=1}^{j=l} (\sqrt{p_j} - \sqrt{\hat{p}_j}}`

        being l the number of classes

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        HD : float
            It is equal to :math:`\sqrt{\sum_{j=1}^{j=l} (\sqrt{p_j} - \sqrt{\hat{p}_j}}`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    dist = np.sqrt(np.sum((np.sqrt(p_pred) - np.sqrt(p_true)) ** 2))
    return dist


def bray_curtis(p_true, p_pred):
    """Bray-Curtis dissimilarity.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sum(np.abs(p_true - p_pred)) / np.sum(p_true + p_pred)


def topsoe(p_true, p_pred, epsilon=1e-20):
    """ Topsoe
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    dist = np.sum(p_true*np.log((2*p_true+epsilon)/(p_true+p_pred+epsilon)) +
                  p_pred*np.log((2*p_pred+epsilon)/(p_true+p_pred+epsilon)))
    return dist


def jensenshannon(p_true, p_pred, epsilon=1e-20):
    """ Jensen-Shannon divergence (a=1/2)
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    dist = np.sum(p_true*np.log(p_true+epsilon)+p_pred*np.log(p_pred+epsilon)-
                  (p_true+p_pred)*np.log((p_true+p_pred+epsilon)/2))/2
    return dist


def probsymmetric(p_true, p_pred, epsilon=1e-20):
    """ Probabilistic Symmetric
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    dist = 2*np.sum((p_true-p_pred)**2 / (p_pred+p_true+epsilon))
    return dist


def brier_multi(targets, probs):
    """ Brier score (classification) for multiclass problems
    """
    return np.mean(np.sum((probs - targets)**2, axis=1))


def geometric_mean(y_true, y_pred, correction=0.0):
    """Compute the geometric mean.

    In quantification, the geometric mean is useful to training a classifier for imbalanced problems (a quite common
    issue). The geometric mean tries to maximize the accuracy for all classes, their accuracies must be balanced to
    obtain a good score for  geometric mean. It is computed as the root of the product of classes sensitivity.

    The optimal value is 1 and the worst is 0 (this occurs when the accuracy for one class is 0). To dealt with
    worst-case for highly multiclass problems, the sensitivity of unrecognized classes can be corrected to a given
    value (instead of zero), see correction parameter.

    The implementation given here is a simplification of the one provide in imbalanced library.

    Parameters
    ----------
    y_true : ndarray, shape (n_examples,)
        True class for each example

    y_pred : array, shape (n_examples,)
        Predicted class returned by a classifier

    correction : float, default=0.0
        Substitutes sensitivity of unrecognized classes from zero to this value.

    Returns
    -------
    g_mean : float
        Returns the geometric mean
    """

    labels = unique_labels(y_true, y_pred)
    n_labels = len(labels)

    le = LabelEncoder()
    le.fit(labels)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    sorted_labels = le.classes_

    tp = y_true == y_pred
    tp_bins = y_true[tp]

    if len(tp_bins):
        tp_sum = np.bincount(tp_bins, weights=None, minlength=len(labels))
    else:
        true_sum = tp_sum = np.zeros(len(labels))

    if len(y_true):
        true_sum = np.bincount(y_true, weights=None, minlength=len(labels))

    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]

    mask = true_sum == 0.0
    true_sum[mask] = 1  # avoid infs/nans
    recall = tp_sum / true_sum
    recall[mask] = 0
    recall[recall == 0] = correction

    with np.errstate(divide="ignore", invalid="ignore"):
        g_mean = scipy.stats.gmean(recall)
    return g_mean


#
#
# def relative_absolute_error(p_true, p_pred, eps=1e-12):
#     """The relation between the absolute error and the true prevalence.
#
#         Parameters
#         ----------
#         p_true : array_like, shape=(n_classes)
#             True prevalences. In case of binary quantification, this parameter could be a single float value.
#
#         p_pred : array_like, shape=(n_classes)
#             Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
#
#         eps: to prevent division by 0
#
#         """
#     if np.any(p_true == 0):
#         return np.mean((np.abs(p_pred - p_true) + eps) / (p_true + eps))
#     return np.mean(np.abs(p_pred - p_true) / p_true)
#
#
# def symmetric_absolute_percentage_error(p_true, p_pred):
#     """SAPE. A symmetric version of RAE.
#
#         Parameters
#         ----------
#         p_true : array_like, shape=(n_classes)
#             True prevalences. In case of binary quantification, this parameter could be a single float value.
#
#         p_pred : array_like, shape=(n_classes)
#             Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
#         """
#     if np.any((p_pred + p_true) == 0):
#         raise NotImplementedError
#     return np.abs(p_pred - p_true) / (p_pred + p_true)
#
#
#
#
# def normalized_absolute_error(p_true, p_pred):
#     """A loss function, ranging from 0 (best) and 1 (worst)
#
#         Parameters
#         ----------
#         p_true : array_like, shape=(n_classes)
#             True prevalences. In case of binary quantification, this parameter could be a single float value.
#
#         p_pred : array_like, shape=(n_classes)
#             Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
#         """
#     return np.sum(np.abs(p_pred - p_true)) / (2 * (1 - np.min(p_true)))
#
#
#
#
# def normalized_relative_absolute_error(p_true, p_pred):
#     """NRAE.
#
#         Parameters
#         ----------
#         p_true : array_like, shape=(n_classes)
#             True prevalences. In case of binary quantification, this parameter could be a single float value.
#
#         p_pred : array_like, shape=(n_classes)
#             Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
#         """
#     l = p_true.shape[0]
#     return relative_absolute_error(p_true, p_pred) / (l - 1 + (1 - np.min(p_true)) / np.min(p_true))

