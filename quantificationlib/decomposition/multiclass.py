"""
Generic quantifiers based on multiclass decompositions
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np
from copy import deepcopy

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

from quantificationlib.base import BaseQuantifier, UsingClassifiers, WithoutClassifiers


class OneVsRestQuantifier(UsingClassifiers):
    """ Implements a One Vs Rest Multiclass Quantifier given any base quantifier

        Trains one quantifier per class that predicts the prevalence on such class. The aggregation strategy
        just normalizes these prevalences to sum 1

        The class works both with quantifiers that require classifiers or not. In the former case, the estimator
        used for the training distribution and the testing distribution must be a OneVsRestQuantifier

        Parameters
        ----------
        base_quantifier : quantifier object
            The base quantifier used to build the One versus Rest decomposition. Any quantifier can be used

        estimator_train : estimator object, optional, (default=None)
            An estimator object implementing `fit` and one of `predict` or `predict_proba`. It is used to classify
            the examples of the training set and to obtain their distribution when the base quantifier is an
            instance of the class UsingClassifiers. Notice that some quantifiers of this kind, namely CC and PCC,
            do not require an estimator for the training distribution

        estimator_test : estimator object, optional, (default=None)
            An estimator object implementing `fit` and one of `predict` or `predict_proba`. It is used to classify
            the examples of the testing bag and to obtain their distribution when the base quantifier is an
            instance of the class UsingClassifiers

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimators could be the same

        Attributes
        ----------
        base_quantifier : quantifier object
            The base quantifier used to build the One versus Rest decomposition

        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        needs_predictions_train : bool, (default=True)
            True if the base quantifier needs to estimate the training distribution

        probabilistic_predictions : bool
            Not used

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilistic)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilistic)
            Predictions of the examples in the testing bag

        quantifiers_ : ndarray, shape (n_classes, )
            List of quantifiers, one for each class

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
    """
    def __init__(self, base_quantifier, estimator_train=None, estimator_test=None, verbose=0):
        super(OneVsRestQuantifier, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                                  verbose=verbose)
        # attributes
        self.base_quantifier = base_quantifier
        self.quantifiers_ = None

    def fit(self, X, y, predictions_train=None):
        """ Fits all the quanfifiers of a OneVsRest decomposition

            First, the method fits the estimators (estimator_train and estimator_test) (if needed) using
            the `fit` method of its superclass

            Then, it creates (using `deepcopy`) the set on quantifiers_, one per class, and fit them

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, optional, shape (n_examples, n_classes) (probs)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train or estimator_test are not instances of OneVsRestClassifier
        """
        if not isinstance(self.base_quantifier, BaseQuantifier):
            raise ValueError("Class %s: quantifier must be an object of a class derived from quantificationlib.baseQuantifier"
                             % self.__class__.__name__)

        #  binarizing y labels
        label_binarized = LabelBinarizer()
        label_binarized.fit(y)
        y_bin = label_binarized.transform(y)

        if isinstance(self.base_quantifier, WithoutClassifiers):

            self.classes_ = np.unique(y)
            if self.verbose > 0:
                print('Class %s: do not have estimators to fit...' % self.__class__.__name__, end='')

            self.quantifiers_ = np.zeros(len(self.classes_), dtype=object)

            for i in range(len(self.classes_)):
                self.quantifiers_[i] = deepcopy(self.base_quantifier)
                self.quantifiers_[i].fit(X, y_bin[:, i])

        elif isinstance(self.base_quantifier, UsingClassifiers):

            if self.estimator_train is not None and not isinstance(self.estimator_train, OneVsRestClassifier):
                raise ValueError("Class %s: estimator_train must be a OneVsRestClassifier" % self.__class__.__name__)

            if self.estimator_test is not None and not isinstance(self.estimator_test, OneVsRestClassifier):
                raise ValueError("Class %s: estimator_test must be a OneVsRestClassifier" % self.__class__.__name__)

            self.needs_predictions_train = self.base_quantifier.needs_predictions_train

            super().fit(X, y, predictions_train=predictions_train)

            self.quantifiers_ = np.zeros(len(self.classes_), dtype=object)
            for i in range(len(self.classes_)):
                self.quantifiers_[i] = deepcopy(self.base_quantifier)
                if self.estimator_train is not None:
                    self.quantifiers_[i].estimator_train = self.estimator_train.estimators_[i]
                else:
                    self.quantifiers_[i].estimator_train = None
                if self.estimator_test is not None:
                    self.quantifiers_[i].estimator_test = self.estimator_test.estimators_[i]
                else:
                    self.quantifiers_[i].estimator_test = None
                if predictions_train is None:
                    self.quantifiers_[i].fit(X, y_bin[:, i])
                else:
                    preds = np.array([1 - predictions_train[:, i], predictions_train[:, i]]).T
                    self.quantifiers_[i].fit(X, y_bin[:, i], predictions_train=preds)
        return self

    def predict(self, X, predictions_test=None):
        """ Aggregates the prevalences of the quantifiers_ to compute the final prediction

            Just one aggregation strategy is implemented. It normalizes the prevalences given by each quantifier
            to sum 1

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                Predictions for the testing bag
        """
        prevalences = np.zeros(len(self.classes_))

        if isinstance(self.base_quantifier, WithoutClassifiers):
            for i in range(len(self.classes_)):
                prevalences[i] = self.quantifiers_[i].predict(X)[1]

        elif isinstance(self.base_quantifier, UsingClassifiers):
            if predictions_test is None:
                for i in range(len(self.classes_)):
                    prevalences[i] = self.quantifiers_[i].predict(X)[1]
            else:
                for i in range(len(self.classes_)):
                    preds = np.array([1 - predictions_test[:, i], predictions_test[:, i]]).T
                    prevalences[i] = self.quantifiers_[i].predict(X, predictions_test=preds)[1]

        if np.sum(prevalences) > 0:
            prevalences = prevalences / float(np.sum(prevalences))

        return prevalences
