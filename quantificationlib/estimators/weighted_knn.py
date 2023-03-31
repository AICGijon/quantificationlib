"""Proportion-weighted K-Nearest Neighbor Classifier"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np
import warnings

from scipy import stats
from sklearn.utils.extmath import weighted_mode

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier


class PWK(BaseEstimator, ClassifierMixin):
    """Proportion-weighted k-Nearest Neighbor Classifier

        This class is an kind of wrapper of sklearn.neighbors.KNeighborsClassifier (version 1.0.2) to use
        class-dependent weights to deal with imbalanced problems. The parameters are the same, except weights
        that are computed by this class

        Parameters
        ----------
        n_neighbors : int, (default=10)
            Number of neighbors to use by default for :meth:`kneighbors` queries.

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
            Algorithm used to compute the nearest neighbors:
            - 'ball_tree' will use :class:`BallTree`
            - 'kd_tree' will use :class:`KDTree`
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
              based on the values passed to :meth:`fit` method.
            Note: fitting on sparse input will override the setting of
            this parameter, using brute force.

        leaf_size : int, default=30
            Leaf size passed to BallTree or KDTree.  This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree.  The optimal value depends on the
            nature of the problem.

        p : int, default=2
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using manhattan_distance (l1), and euclidean_distance
            (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

        metric : str or callable, default='minkowski'
            The distance metric to use for the tree.  The default metric is
            minkowski, and with p=2 is equivalent to the standard Euclidean
            metric. For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.
            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square during fit. X may be a :term:`sparse graph`,
            in which case only "nonzero" elements may be considered neighbors.

        metric_params : dict, default=None
            Additional keyword arguments for the metric function.

        n_jobs : int, default=None
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
            Doesn't affect :meth:`fit` method.

        Attributes
        ----------
        knn_ : KNeighborsClassifier object
            KNN classifier

        classes_ : array of shape (n_classes,)
            Class labels known to the classifier

        weights_ : array, shape (n_samples, )
            The weight for each example

        y_ : array
            True labels

    """
    def __init__(self, n_neighbors=10, algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None,
                 n_jobs=None):
        self.knn_ = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size,
                                         metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)
        self.classes_ = None
        self.weights_ = None
        self.y_ = None

    def fit(self, X, y):
        """ Fit the k-nearest neighbors classifier and compute the weights using the training dataset

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes
        """
        self.y_ = y
        self.classes_ = np.unique(y)
        self.weights_ = np.ones(X.shape[0])
        for n_cls, cls in enumerate(self.classes_):
            self.weights_[y == cls] = 1 - (np.sum(self.y_ == cls) / X.shape[0])

        self.knn_.fit(X, y)
        return self

    def predict(self, X):
        """ Returns the crisp predictions for the provided data

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Test ata

            Returns
            -------
            preds : array-like, shape shape(n_examples, )
                 Crisp predictions for the examples in X
        """
        _, neigh_ind = self.knn_.kneighbors(X)

        mode, _ = weighted_mode(self.y_[neigh_ind], self.weights_[neigh_ind], axis=1)
        y_pred = np.asarray(mode.ravel())
        return y_pred

    def predict_proba(self, X):
        """ Returns the probabilistic predictions for the provided data

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Test ata

            Returns
            -------
            preds : array-like, shape shape(n_examples, n_classes)
                 Probabilistic predictions for the examples in X
        """
        _, neigh_ind = self.knn_.kneighbors(X)

        probabilities = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(X.shape[0]):
            kneigh_classes = self.y_[neigh_ind[i, :]]
            for n_cls, cls in enumerate(self.classes_):
                probabilities[i, n_cls] = np.sum(self.weights_[neigh_ind[i, kneigh_classes == cls]])

        # compute actual probabilities given the weights for each class
        normalizer = probabilities.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        probabilities /= normalizer

        return probabilities
