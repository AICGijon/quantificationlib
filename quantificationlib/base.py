"""
Base classes for all quantifiers
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np
import six
from abc import ABCMeta

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_X_y, check_array


class BaseQuantifier(six.with_metaclass(ABCMeta, BaseEstimator)):
    """ Base class for binary, multiclass and ordinal quantifiers
    """
    pass


class WithoutClassifiers(BaseQuantifier):
    """ Base class for quantifiers that do not use any classifier

        Examples of this type of quantifiers are HDX and EDX, for instance

        Parameters
        ----------
         verbose : int, optional, (default=0)
             The verbosity level. The default value, zero, means silent mode

         Attributes
         ----------
         verbose : int
             The verbosity level

         classes_ : ndarray, shape (n_classes, )
             Class labels
     """
    def __init__(self, verbose=0, **kwargs):
        super(WithoutClassifiers, self).__init__(**kwargs)
        # init attributes
        self.verbose = verbose
        # computed attributes
        self.classes_ = None

    def fit(self, X, y):
        """ This method just checks X and Y and stores the classes of the datasets

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes
        """
        check_X_y(X, y, accept_sparse=True)
        self.classes_ = np.unique(y)


class UsingClassifiers(BaseQuantifier):
    """ Base class for quantifiers based on the use of classifiers

        Classes derived from this abstract class work in two different ways:

        1) Two estimators are used to classify the examples of the training set and the testing set in order to
           compute the distributions of both sets. Estimators can be already trained.

        2) You can directly provide the predictions for the examples in the `fit`/`predict` methods. This is useful
           for synthetic/artificial experiments.

        The idea in both cases is to guarantee that all methods based on classifiers are using **exactly**
        the same predictions when you compare this kind of quantifiers. In the first case, estimators are only
        trained once and can be shared for several quantifiers of this kind.

        This class is responsible for fitting the estimators (when needed) and for computing the predictions for the
        training set and the testing set.

        Parameters
        ----------
        estimator_train : estimator object, optional, (default=None)
            An estimator object implementing `fit` and one of `predict` or `predict_proba`. It is used to classify
            the examples of the training set and to obtain their distribution

        estimator_test : estimator object, optional, (default=None)
            An estimator object implementing `fit` and one of `predict` or `predict_proba`. It is used to classify
            the examples of the testing bag and to obtain their distribution

        needs_predictions_train : bool, (default=True)
            True if the quantifier needs to estimate the training distribution

        probabilistic_predictions : bool, optional, (default=True)
            Whether the estimators return probabilistic predictions or not. This depends on the specific quantifier,
            some need crisp predictions (e.g. CC) and other methods require probabilistic predictions (PAC, HDy, ...)

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimators could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        needs_predictions_train : bool
            True if the quantifier needs to estimate the training distribution

        probabilistic_predictions : bool
            Whether the estimators return probabilistic predictions or not

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilistic)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilistic)
            Predictions of the examples in the testing bag

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_), )
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying
            CV_estimator) than y. In other cases, y_ext_ == y.
            y_ext_ must be used in `fit`/`predict` methods whenever the true labels of the training set are needed,
            instead of y

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train_/predictions_train and estimator_test_/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used
    """

    def __init__(self, estimator_train=None, estimator_test=None, needs_predictions_train=True,
                 probabilistic_predictions=True, verbose=0, **kwargs):
        super(UsingClassifiers, self).__init__(**kwargs)
        # init attributes
        self.estimator_train = estimator_train
        self.estimator_test = estimator_test
        self.needs_predictions_train = needs_predictions_train
        self.probabilistic_predictions = probabilistic_predictions
        self.verbose = verbose
        # computed attributes
        self.predictions_test_ = None
        self.predictions_train_ = None
        self.classes_ = None
        self.y_ext_ = None

    def fit(self, X, y, predictions_train=None):
        """ Fits the estimators (estimator_train and estimator_test) and computes the predictions for the training
            set (predictions_train_ attribute)

            First, the method checks that estimator_train and predictions_train are not both None

            Then, it fits both estimators if needed. It checks whether the estimators are already trained or not
            by calling the `predict` method.

            The method finally computes predictions_train_ (if needed, attribute needs_predictions_train) using
            predictions_train or estimator_train. If predictions_train is not None, predictions_train_ is copied from
            predictions_train (and converted to crisp values, using `__probs2crisps` method, when
            probabilistic_predictions is False). If predictions_train is None, predictions_train_ is computed using
            the `predict`/`predict_proba` method of estimator_train, depending again on the value of
            probabilistic_predictions attribute.

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, optional, shape(n_examples, 1) crisp or shape (n_examples, n_classes) (probs)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        self.classes_ = np.unique(y)

        if self.needs_predictions_train and self.estimator_train is None and predictions_train is None:
            raise ValueError("estimator_train or predictions_train must be not None "
                             "with objects of class %s", self.__class__.__name__)

        # Fit estimators if they are not already fitted
        if self.estimator_train is not None:
            if self.verbose > 0:
                print('Class %s: Fitting estimator for training distribution...' % self.__class__.__name__, end='')
            # we need to fit the estimator for the training distribution
            # we check if the estimator is trained or not
            try:
                self.estimator_train.predict(X[0:1, :].reshape(1, -1))
                if self.verbose > 0:
                    print('it was already fitted')

            except NotFittedError:

                X, y = check_X_y(X, y, accept_sparse=True)

                self.estimator_train.fit(X, y)

                if self.verbose > 0:
                    print('fitted')

        if self.estimator_test is not None:
            if self.verbose > 0:
                print('Class %s: Fitting estimator for testing distribution...' % self.__class__.__name__, end='')

            # we need to fit the estimator for the testing distribution
            # we check if the estimator is trained or not
            try:
                self.estimator_test.predict(X[0:1, :].reshape(1, -1))
                if self.verbose > 0:
                    print('it was already fitted')

            except NotFittedError:

                X, y = check_X_y(X, y, accept_sparse=True)

                self.estimator_test.fit(X, y)

                if self.verbose > 0:
                    print('fitted')

        # Compute predictions_train_
        if self.verbose > 0:
            print('Class %s: Computing predictions for training distribution...' % self.__class__.__name__, end='')

        if self.needs_predictions_train:
            if predictions_train is not None:
                if self.probabilistic_predictions:
                    self.predictions_train_ = predictions_train
                else:
                    self.predictions_train_ = UsingClassifiers.__probs2crisps(predictions_train, self.classes_)
            else:
                if self.probabilistic_predictions:
                    self.predictions_train_ = self.estimator_train.predict_proba(X)
                else:
                    self.predictions_train_ = self.estimator_train.predict(X)

            # Compute y_ext_
            if len(y) == len(self.predictions_train_):
                self.y_ext_ = y
            else:
                self.y_ext_ = np.tile(y, len(self.predictions_train_) // len(y))

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Computes the predictions for the testing set (predictions_test_ attribute)

            First, the method checks if at least one between estimator_test and prediction_test is not None,
            otherwise a ValueError exception is raised.

            Then, it computes predictions_test_. If predictions_test is not None, predictions_test_ is copied from
            predictions_test (and converted to crisp values, using `__probs2crisp` method when
            probabilistic_predictions attribute is False). If predictions_test is None, predictions_test_ is computed
            calling the `predict`/`predict_proba method (depending on the value of the attribute
            probabilistic_predictions) of estimator_test.

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                Predictions for the testing bag

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are both None
        """
        if self.estimator_test is None and predictions_test is None:
            raise ValueError("estimator_test or predictions_test must be not None "
                             "to compute a prediction with objects of class %s", self.__class__.__name__)

        if self.verbose > 0:
            print('Class %s: Computing predictions for testing distribution...' % self.__class__.__name__, end='')

        # At least one between estimator_test and predictions_test is not None
        if predictions_test is not None:
            if self.probabilistic_predictions:
                self.predictions_test_ = predictions_test
            else:
                self.predictions_test_ = UsingClassifiers.__probs2crisps(predictions_test, self.classes_)
        else:
            check_array(X, accept_sparse=True)
            if self.probabilistic_predictions:
                self.predictions_test_ = self.estimator_test.predict_proba(X)
            else:
                self.predictions_test_ = self.estimator_test.predict(X)

        if self.verbose > 0:
            print('done')

        return self

    @staticmethod
    def __probs2crisps(preds, labels):
        """ Convert probability predictions to crisp predictions

            Parameters
            ----------
            preds : ndarray, shape (n_examples, 1) or (n_examples,) for binary problems, (n_examples, n_classes) for
                    multiclass
                The matrix with the probability predictions

            labels : ndarray, shape (n_classes, )
                Class labels
        """
        if len(preds) == 0:
            return preds
        if preds.ndim == 1 or preds.shape[1] == 1:
            #  binary problem
            if preds.ndim == 1:
                preds_mod = np.copy(preds)
            else:
                preds_mod = np.copy(preds.squeeze())
            if isinstance(preds_mod[0], np.float):
                # it contains probs
                preds_mod[preds_mod >= 0.5] = 1
                preds_mod[preds_mod < 0.5] = 0
                return preds_mod.astype(int)
            else:
                return preds_mod
        else:
            # multiclass problem
            return labels.take(preds.argmax(axis=1), axis=0)
