"""
PWKQuantifier a quantifier based on K-Nearest Neighbor
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from sklearn.metrics import confusion_matrix

from quantificationlib.base import WithoutClassifiers
from quantificationlib.estimators.weighted_knn import PWK
from quantificationlib.optimization import solve_l1


class PWKQuantifier(WithoutClassifiers):
    """ Quantifier based on K-Nearest Neighbor proposed by (Barranquero et al., 2013)

        It is a AC method in which the estimator is PWK, a weighted version of KNN in
        which the weight depends on the proportion of each class in the training set. It is not derived from AC to
        allow decomposition

        Parameters
        ----------
        n_neighbors : int, (default=10)
            Number of neighbors to use by default for :meth:`kneighbors` queries.
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

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        classes_ : ndarray, shape (n_classes, )
             Class labels

        cm_ : ndarray, shape (n_classes, n_classes)
            Confusion matrix. The true classes are in the rows and the predicted classes in the columns. So, for
            the binary case, the count of true negatives is cm_[0,0], false negatives is cm_[1,0],
            true positives is cm_[1,1] and false positives is cm_[0,1] .

        problem_ : a cvxpy Problem object
            This attribute is set to None in the fit() method. With such model, the first time a testing bag is
            predicted this attribute will contain the corresponding cvxpy Object (if such library is used, i.e in the
            case of 'L1' and 'HD'). For the rest testing bags, this object is passed to allow a warm start. The
            solving process is faster.

        verbose : int
            The verbosity level

        References
        ----------
        Jose Barranquero, Pablo González, Jorge Díez, Juna José del Coz: On the study of nearest neighbor algorithms
        for prevalence estimation in binary problems. Pattern Recognition, 46(2), 472-482. 2013
    """
    def __init__(self, n_neighbors=10, p=2, metric='minkowski', metric_params=None, verbose=0):
        super(PWKQuantifier, self).__init__(verbose=verbose)
        #  Attributes
        self.knn_estimator_ = PWK(n_neighbors=n_neighbors, p=p, metric=metric, metric_params=metric_params)
        self.cm_ = None
        self.problem_ = None

    def fit(self, X, y):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (crisp values) if needed. Both operations are
            performed by the `fit` method of its superclass.
            Finally the method computes the confusion matrix of the training set using predictions_train_

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes
        """
        super().fit(X, y)

        self.knn_estimator_.fit(X, y)
        predictions_train = self.knn_estimator_.predict(X)

        if self.verbose > 0:
            print('Class %s: Estimating confusion matrix for training distribution...' % self.__class__.__name__,
                  end='')

        #  estimating the confusion matrix
        cm = confusion_matrix(y, predictions_train, labels=self.classes_)
        #  normalizing cm by row
        self.cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # binary:  [[1-fpr  fpr]
        #                                                                          [1-tpr  tpr]]

        if self.verbose > 0:
            print('done')

        self.problem_ = None

        return self

    def predict(self, X):
        """ Predict the class distribution of a testing bag

            The prevalences are computed solving a system of linear scalar equations:

                         cm_.T * prevalences = CC(X)

            For binary problems the system is directly solved using the original AC algorithm proposed by Forman

                        p = (p_0 - fpr ) / ( tpr - fpr)

            For multiclass problems, the system may not have a solution. Thus, instead we propose to solve an
            optimization problem of this kind:

                      Min   distance ( cm_.T * prevalences, CC(X) )
                      s.t.  sum(prevalences) = 1
                            prevalecences_i >= 0

            in which distance is 'L1'

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        predictions_test = self.knn_estimator_.predict(X)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)

        freq = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls, 0] = np.equal(predictions_test, cls).sum()

        prevalences_0 = freq / float(len(predictions_test))

        if n_classes == 2:
            if np.abs((self.cm_[1, 1] - self.cm_[0, 1])) > 0.001:
                p = (prevalences_0[1] - self.cm_[0, 1]) / (self.cm_[1, 1] - self.cm_[0, 1])
                prevalences = [1-p, p]
            else:
                prevalences = prevalences_0

            # clipping the prevalences according to (Forman 2008)
            prevalences = np.clip(prevalences, 0, 1)

            if np.sum(prevalences) > 0:
                prevalences = prevalences / float(np.sum(prevalences))

            prevalences = prevalences.squeeze()
        else:
            self.problem_, prevalences = solve_l1(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                                  n_classes=n_classes, problem=self.problem_)

        if self.verbose > 0:
            print('done')

        return prevalences
