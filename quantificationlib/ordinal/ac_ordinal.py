"""
Ordinal version of AC quantifier
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from sklearn.metrics import confusion_matrix

from quantificationlib.base import UsingClassifiers
from quantificationlib.optimization import solve_l1, solve_hd


class ACOrdinal(UsingClassifiers):
    """ Adjusted Count method for Ordinal Quantification

        The idea is to compute the prevalences that minimizes the EMD distance between cm * prevalences (cm is the
        confusion matrix) and the prevalences observed in the testing distribution using the CC method.

        This class works in two different ways:

        1) Two estimators are used to classify the examples of the training set and the testing set in order to
           compute the confusion matrix of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the `fit`/`predict methods. This is useful
           for synthetic/artificial experiments

        The idea in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, PDFyOrdinal). In the first case, estimators are only trained once and can be shared
        for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict`. It is used to classify the examples of the training
            set and to compute the confusion matrix

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict`. It is used to classify the examples of the testing
            set and to obtain their predictions

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimators could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, ) (crisp estimator)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, ) (crisp estimator)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because AC quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, False
             This means that predictions_test_ contains crisp predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit` method whenever the true labels of the training set are needed, instead of y

        cm_ : ndarray, shape (n_classes, n_classes)
            Confusion matrix

        train_distrib_ : ndarray, shape (n_classes_, n_classes)
            The cumulative distribution for each class in the training set

        test_distrib_ : ndarray, shape (n_classes_, 1)
            The cumulative distribution for the testing bag

        problem_ : a cvxpy Problem object
            This attribute is set to None in the fit() method. With such model, the first time a testing bag is
            predicted this attribute will contain the corresponding cvxpy Object (if such library is used, i.e in the
            case of 'L1' and 'HD'). For the rest testing bags, this object is passed to allow a warm start. The
            solving process is faster.

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        George Forman. 2008. Quantifying counts and costs via classification. Data Mining Knowledge Discovery 17,
        2 (2008), 164–206.
    """

    def __init__(self, estimator_train=None, estimator_test=None, verbose=0):
        super(ACOrdinal, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                        needs_predictions_train=True, probabilistic_predictions=False, verbose=verbose)
        # confusion matrix
        self.cm_ = None
        self.train_distrib_ = None
        self.test_distrib_ = None
        self.problem_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (crisp values) if needed. Both operations are
            performed by the `fit` method of its superclass.
            Finally the method computes the confusion matrix of the training set using predictions_train_, and
            the training distribution

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, ) or (n_examples, n_classes)
                Predictions of the examples in the training set. If shape is (n_examples, n_classes) predictions are
                converted to crisp values by `super().fit()

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating confusion matrix for training distribution...' % self.__class__.__name__,
                  end='')

        #  estimating the confusion matrix
        cm = confusion_matrix(self.y_ext_, self.predictions_train_, labels=self.classes_)
        #  normalizing cm by row
        self.cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        self.train_distrib_ = np.array(self.cm_.T, copy=True)
        n_classes = len(self.classes_)
        for i in range(1, n_classes):
            self.train_distrib_[i, :] = np.sum(self.train_distrib_[i - 1:i + 1, :], axis=0)

        if self.verbose > 0:
            print('done')

        self.problem_ = None

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the distribution of such predictions are computed and stored in test_distrib_ attribute
            Finally, the prevalences are computed solving the following optimization problem:

                      Min   | train_distrib_ * prevalences -  test_distrib_ |
                      s.t.  sum(prevalences) = 1
                            prevalecences_i >= 0

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba method)

                If predictions_test is not None they are copied on predictions_test_ and used.
                If predictions_test is None, predictions for the testing examples are computed using the `predict`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are both None

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)

        freq = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls, 0] = np.equal(self.predictions_test_, cls).sum()

        prevalences_0 = freq / float(len(self.predictions_test_))

        self.test_distrib_ = np.array(prevalences_0, copy=True)
        for i in range(1, n_classes):
            self.test_distrib_[i] = self.test_distrib_[i] + self.test_distrib_[i - 1]

        self.problem_, prevalences = solve_l1(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                              n_classes=n_classes, problem=self.problem_)

        if self.verbose > 0:
            print('done')

        return prevalences
