"""
Score functions and loss functions for binary quantification problems
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np


def binary_kld(p_true, p_pred, eps=1e-12):
    """ A binary version of the Kullback - Leiber divergence (KLD)

            :math:`kld = p \cdot \log(p/\hat{p}) + (1-p) \cdot \log((1-p)/(1-\hat{p}))

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences.

        eps : float, (default=1e-12)
            To prevent a division by zero exception

        Returns
        -------
        KLD: float
            It is equal to :math:`p \cdot \log(p/\hat{p}) + (1-p) \cdot \log((1-p)/(1-\hat{p}))`
    """
    if p_pred == 0:
        kld = p_true * np.log2(p_true / eps)
    else:
        kld = p_true * np.log2(p_true / p_pred)
    if p_pred == 1:
        kld = kld + (1 - p_true) * np.log2((1 - p_true) / eps)
    else:
        kld = kld + (1 - p_true) * np.log2((1 - p_true) / (1 - p_pred))
    return kld


def bias(p_true, p_pred):
    """ Bias of a binary quantifier

        It is just the difference between the predicted prevalence (:math:`\hat{p}`) and the true prevalence (:math:`p`)

            :math:`bias = \hat{p} - p`

        It measures whether the binary quantifier tends to overestimate or underestimate the proportion of positives

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        Returns
        -------
        bias: float
            It is equal to :math:`\hat{p} - p`
    """
    return p_pred - p_true


def absolute_error(p_true, p_pred):
    """ Binary version of the absolute error

        Absolute difference between the predicted prevalence (:math:`\hat{p}`) and the true prevalence (:math:`p`)

            :math:`ae = | \hat{p} - p |`

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        Returns
        -------
        absolute error: float
            It is equal to :math:`| \hat{p} - p |`
    """
    return np.abs(p_pred - p_true)


def squared_error(p_true, p_pred):
    """ Binary version of the squared error. Only the prevalence of the positive class is used

        It is the quadratic difference between the predicted prevalence (:math:`\hat{p}`) and
        the true prevalence (:math:`p`)

            :math:`se = (\hat{p} - p)^2`

        It penalizes larger errors

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        Returns
        -------
        squared_error: float
            It is equal to :math:`(\hat{p} - p)^2`
    """
    return (p_pred - p_true) ** 2


def relative_absolute_error(p_true, p_pred, eps=1e-12):
    """ A binary relative version of the absolute error

        It is the relation between the absolute error and the true prevalence.

            :math:`rae = | \hat{p} - p | / p`

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        eps : float, (default=1e-12)
            To prevent a division by zero exception

        Returns
        -------
        relative_absolute_error: float
            It is equal to :math:`| \hat{p} - p | / p`
    """
    if p_true == 0:
        return np.abs(p_pred - p_true) / (p_true + eps)
    else:
        return np.abs(p_pred - p_true) / p_true


def symmetric_absolute_percentage_error(p_true, p_pred):
    """ A symmetric binary version of RAE

            :math:`sape = | \hat{p} - p | / (\hat{p} + p)`

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        Returns
        -------
        SAPE: float
            It is equal to :math:`| \hat{p} - p | / (\hat{p} + p)`
    """
    if p_pred + p_true == 0:
        return 0
    else:
        return np.abs(p_pred - p_true) / (p_pred + p_true)


def normalized_absolute_score(p_true, p_pred):
    """ A score version of the normalized binary absolute error

            :math:`nas = 1 - | \hat{p} - p | / max(p, 1-p)`

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        Returns
        -------
        NAS: float
            It is equal to :math:`1 - | \hat{p} - p | / max(p, 1-p)`
    """
    return 1 - np.abs(p_pred - p_true) / np.max([p_true, 1 - p_true])


def normalized_squared_score(p_true, p_pred):
    """ A score version of the normalized binary squared error

            :math:`nss = 1 - ( (\hat{p} - p) / max(p, 1-p) )^2`

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        Returns
        -------
        NSS: float
            It is equal to :math:`1 - ( (\hat{p} - p) / max(p, 1-p) )^2`
    """
    return 1 - (np.abs(p_pred - p_true) / np.max([p_true, 1 - p_true])) ** 2
