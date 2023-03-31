"""
Multiclass versions for CC and PCC quantifiers
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from quantificationlib.base import UsingClassifiers


class CC(UsingClassifiers):
    """ Multiclass Classify And Count method

        prevalence (class_i) = (1/|Test|) *  sum_{x in Test} I ( h(x) == class_i)

        This class works in two different ways:

        1) An estimator is used to classify the examples of the testing bag (the estimator can be already trained)

        2) You can directly provide the predictions for the examples in the predict method. This is useful
           for synthetic/artificial experiments

        Parameters
        ----------
        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict` methods

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        estimator_train : None. (Not used)

        estimator_test : estimator object
            Estimator used to classify the examples of the testing bag

        needs_predictions_train : bool, False
            It is False because CC quantifiers do not need to estimate the training distribution

        probabilistic_predictions : bool, False
             This means that predictions_test_ contains crisp predictions

        predictions_test_ : ndarray, shape (n_examples, )
            Crisp predictions of the examples in the testing bag

        predictions_train_ : None. (Not used)

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(n_examples, )
            True labels of the training set

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_test and predictions_test must be not None. If both are None a
        ValueError exception will be raised. If both are not None, predictions_test is used.

        References
        ----------
        George Forman. 2005. Counting positives accurately despite inaccurate classification. In Proceedings of
        the European Conference on Machine Learning (ECML’05). 564–575.

        George Forman. 2008. Quantifying counts and costs via classification. Data Mining Knowledge Discovery 17,
        2 (2008), 164–206.
    """
    def __init__(self, estimator_test=None, verbose=0):
        super(CC, self).__init__(estimator_test=estimator_test,
                                 needs_predictions_train=False, probabilistic_predictions=False, verbose=verbose)

    def fit(self, X, y, predictions_train=None):
        """ Fit the estimator for the testing bags when needed. The method checks whether the estimator is trained or
            not calling the predict method

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            y : (sparse) array-like, shape (n_examples, )
                True classes

            predictions_train : None, not used
                Predictions of the examples in the training set.
        """
        super().fit(X, y, predictions_train=[])

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            The prevalence for each class is the proportion of examples predicted as belonging to that class

            prevalence (class_i) = (1/|Test|) *  sum_{x in Test} I ( h(x) == class_i)

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            predictions_test : ndarray, shape (n_examples, ) or (n_examples, n_classes) (default=None)
                They can be crisp values or probabilities. In the latter case, they are converted to crisp values
                using `__probs2crisps` method

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

        n_classes = len(self.classes_)

        freq = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls, 0] = np.equal(self.predictions_test_, cls).sum()

        prevalences = freq / float(len(self.predictions_test_))

        if self.verbose > 0:
            print('done')

        return np.squeeze(prevalences)


class PCC(UsingClassifiers):
    """ Multiclass Probabilistic Classify And Count method.

        prevalence (class_i) = sum_{x in T} P( h(x) == class_i | x )

        This class works in two different ways:

        1) An estimator is used to classify the examples of the testing bag (the estimator can be already trained)

        2) You can directly provide the predictions for the examples in the predict method. This is useful
           for synthetic/artificial experiments

        Parameters
        ----------
        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict` methods. It is used to classify the testing examples

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_test_ : ndarray, shape (n_examples, n_classes)
            Probabilistic predictions of the examples in the testing bag

        estimator_train : None. (Not used)

        predictions_train_ : None. (Not used)

        needs_predictions_train : bool, False
            It is False because PCC quantifiers do not need to estimate the training distribution

        probabilistic_predictions : bool, True
             This means that predictions_test_ contains probabilistic predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(n_examples, )
            True labels of the training set

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_test and predictions_test must be not None. If both are None a
        ValueError exception will be raised. If both are not None, predictions_test is used.

        References
        ----------
        Antonio Bella, Cèsar Ferri, José Hernández-Orallo, and María José Ramírez-Quintana. 2010. Quantification
        via probability estimators. In Proceedings of the IEEE International Conference on Data Mining (ICDM’10).
        IEEE, 737–742.
    """
    def __init__(self, estimator_test=None, verbose=0):
        super(PCC, self).__init__(estimator_test=estimator_test,
                                  needs_predictions_train=False, probabilistic_predictions=True, verbose=verbose)

    def fit(self, X, y, predictions_train=None):
        """ Fit the estimator for the testing bags when needed. The method checks whether the estimator is trained or
            not calling the predict method

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            y : (sparse) array-like, shape (n_examples, )
                True classes

            predictions_train : Not used
        """
        super().fit(X, y, predictions_train=[])

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            The prevalence for each class is the average probability for such class

            prevalence (class_i) = sum_{x in T} P( h(x) == class_i | x )

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

        prevalences = np.mean(self.predictions_test_, axis=0)

        if self.verbose > 0:
            print('done')

        return prevalences
