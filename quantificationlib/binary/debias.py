"""
De-Bias quantifier (just for binary quantification)
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from quantificationlib.base import UsingClassifiers


class DeBias(UsingClassifiers):
    """ Binary quantifier based on De-Bias estimate proposed by Friedman

        prevalence (positives) = prior(positives)  + ( prevalence_PCC - prior(positives) ) / Vt

        where

        Vt =[ 1/|T| sum_{x in D} (P(h(x)==+1|x) - prior(positives) )^2 ] / (prior(positives) * prior(negatives))

        This class works in two different ways:

        1) An estimator is used to classify the examples of the testing bag (the estimator can be already trained)

        2) You can directly provide the predictions for the examples in the predict method. This is useful
           for synthetic/artificial experiments

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the confusion matrix

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to obtain the confusion matrix of the testing set

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimators could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set.

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the testing bag

        probabilistic_predictions : bool, True
             This means that predictions_train_/predictions_test_ contain probabilistic predictions

        needs_predictions_train : bool, True
            It is True because DeBias quantifiers need to estimate the training distribution

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(n_examples, )
            True labels of the training set

        train_prevs_ : ndarray, shape (n_classes, )
            Prevalence of each class in the training set

        Vt_ : float
           The value of equation
                Vt =[ 1/|T| sum_{x in D} (P(h(x)==+1|x) - train_prevs_[1])^2 ] / (train_prevs_[1] * train_prevs_[0])
           applied over the training examples D

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Jerome Friedman. Class counts in future unlabeled samples. Presentation at MIT CSAIL Big Data Event, 2014.
    """

    def __init__(self, estimator_test=None, estimator_train=None, verbose=0):
        super(DeBias, self).__init__(estimator_test=estimator_test, estimator_train=estimator_train,
                                     needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # priors
        self.train_prevs_ = None
        # Vt value
        self.Vt_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit method of its superclass.

            Finally the method computes the value of Vt

            Vt =[ 1/|T| sum_{x in D} (P(h(x)==+1|x) - prior(positives) )^2 ] / (prior(positives) * prior(negatives))

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
            AttributeError
                When the number of classes > 2
        """
        if len(np.unique(y)) != 2:
            raise AttributeError("DB is a binary method, multiclass quantification is not supported")

        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating Vt for training distribution...' % self.__class__.__name__, end='')

        self.train_prevs_ = np.unique(y, return_counts=True)[1] / len(y)

        At = np.mean((self.predictions_train_[:, 1] - self.train_prevs_[1]) ** 2)
        self.Vt_ = At / (self.train_prevs_[1] * self.train_prevs_[0])

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            The prevalence for the positive class is

            prevalence (positives) = prior(positives)  + ( prevalence_PCC - prior(positives) ) / Vt

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a predict_proba method)

                If predictions_test is not None they are copied on predictions_test_ and used.
                If predictions_test is None, predictions for the testing examples are computed using the `predict`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are both None

            Returns
            -------
            prevalences : An ndarray, shape(n_classes, ) with the prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        p = self.train_prevs_[1] + (np.mean(self.predictions_test_, axis=0)[1] - self.train_prevs_[1]) / self.Vt_
        prevalences = np.array([1 - p, p])

        prevalences = np.clip(prevalences, 0, 1)

        if np.sum(prevalences) > 0:
            prevalences = prevalences / float(np.sum(prevalences))

        if self.verbose > 0:
            print('done')

        return prevalences
