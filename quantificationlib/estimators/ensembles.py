"""
Estimators for an Ensemble of Quantifiers
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo


import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.exceptions import NotFittedError

from joblib import Parallel, delayed


class EnsembleOfClassifiers(BaseEstimator, ClassifierMixin):
    """ Ensemble of Classifiers

        This kind of objects train the set of classifiers for an ensemble of quantifiers

        Parameters
        ----------
        base_estimator : estimator object (default=None)
            An estimator object implementing `fit` and one of `predict` or `predict_proba`. It is the base estimator
            used to learn the set of classifiers

        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        base_estimator : estimator object
            The base estimator used to build ensemble

        n_jobs : int or None,
            The number of jobs to use for the computation.

        verbose : int
            The verbosity level. The default value, zero, means silent mode

        classes_ : ndarray, shape (n_classes, )
            Class labels

        n_estimators_ : int,
            Number of estimators

        estimators_ : ndarray, shape(n_ensembles,)
            List of estimators

        References
        ----------
        Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017). Using ensembles for problems with characterizable
        changes in data distribution: A case study on quantification. Information Fusion, 34, 87-100.

        Pérez-Gállego, P., Castano, A., Quevedo, J. R., & del Coz, J. J. (2019). Dynamic ensemble selection
        for quantification tasks. Information Fusion, 45, 1-15.
    """

    def __init__(self, base_estimator=None, n_jobs=None, verbose=0):
        self.base_estimator = base_estimator
        self.verbose = verbose
        self.n_jobs = n_jobs
        # computed variables
        self.classes_ = None
        self.n_estimators_ = 0
        self.estimators_ = None

    def fit(self, X, y, indexes):
        """ Fits the set of estimators for the training set (X, y) using the bags contained in indexes

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            y : (sparse) array-like, shape (n_examples, )
                True classes

            indexes: array-like, shape (n_ensembles, bag_size)
                i-th row contains the indexes of the examples in (X, y) that must be used to train i-th estimator

            Raises
            ------
            ValueError
                When base_estimator is None
        """
        if self.classes_ is None and self.estimators_ is None:

            #  estimators_ are not fitted
            if self.base_estimator is None:
                raise ValueError("An estimator is needed for %s objects", self.__class__.__name__)

            X, y = check_X_y(X, y, accept_sparse=True)

            self.n_estimators_ = indexes.shape[1]
            # fit the estimator for each binary combination of classes, n_estimators = n_classes -1
            self.estimators_ = np.empty(self.n_estimators_, dtype=object)

            # In cases where individual estimators are very fast to train, setting n_jobs > 1 can result in slower
            # performance due to the overhead of spawning threads.  See joblib issue #112.
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(self.base_estimator.fit)(X[indexes[:, n_est], :],
                                                                                             y[indexes[:, n_est]])
                                                            for n_est in range(self.n_estimators_))

        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """ Predict the class for each testing example applying each estimator

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            Returns
            -------
            An array, shape(n_examples, n_estimators) with the predicted class for each example with each estimator

            Raises
            ------
            NotFittedError
                When the estimators are not fitted yet
        """
        if self.classes_ is None:
            raise NotFittedError("This instance of %s class is not fitted yet", type(self).__name__)

        #  we obtain the predictions using the first estimator. We do this because the number of predictions could be
        #  different than len(X)
        p_0 = self.estimators_[0].predict(X)
        preds = np.zeros((len(p_0), self.n_estimators_), dtype=p_0.dtype)
        preds[:, 0] = p_0
        for i in range(1, self.n_estimators_):
            preds[:, i] = self.estimators_[i].predict(X)
        return preds

    def predict_proba(self, X):
        """ Predict the class probabilities for each example `

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            Returns
            -------
            An array, shape(n_examples, n_estimators, n_classes) with the predicted class for each example with each
            estimator

            Raises
            ------
            NotFittedError
                When the estimators are not fitted yet
        """
        if self.classes_ is None:
            raise NotFittedError("This instance of %s class is not fitted yet", type(self).__name__)

        #  we obtain the predictions using the first estimator. We do this because the number of predictions could be
        #  different than len(X)
        p_0 = self.estimators_[0].predict_proba(X)
        preds = np.zeros((len(p_0), self.n_estimators_, len(self.classes_)), dtype=p_0.dtype)
        preds[:, 0, :] = p_0
        for i in range(self.n_estimators_):
            preds[:, i, :] = self.estimators_[i].predict_proba(X)
        return preds
