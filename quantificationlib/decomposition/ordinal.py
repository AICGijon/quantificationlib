"""
Generic decomposition quantifier based on Frank and Hall approach
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np
from copy import deepcopy

from quantificationlib.base import BaseQuantifier, UsingClassifiers, WithoutClassifiers
from quantificationlib.estimators.frank_and_hall import FrankAndHallClassifier, FHLabelBinarizer


class FrankAndHallQuantifier(UsingClassifiers):
    """ Implements a Frank and Hall Ordinal Quantifier given any base quantifier

        Trains one quantifier per each model of the Frank and Hall (FH) decocompositon. For instance, in a ordinal
        classification problem with classes ranging from 1-star to 5-star, FHQuantifier trains 4 quantifiers:
        1 vs 2-3-4-5, 1-2 vs 3-4-5, 1-2-3 vs 4-5, 1-2-3-4 vs 5 and combines their predictions. The positive class
        correspond to the left group of each quantifier ({1}, {1,2}, and so on)

        The class works both with quantifiers that require classifiers or not. In the former case, the estimator
        used for the training distribution and the testing distribution must be a FrankAndHallClassifier

        Parameters
        ----------
        quantifier : quantifier object
            The base quantifier used to build the FH decomposition. Any quantifier can be used

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
        quantifier : quantifier object
            The base quantifier used to build the FH decomposition

        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes-1) (probabilistic)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes-1) (probabilistic)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, (default=True)
            True if the base quantifier needs to estimate the training distribution

        probabilistic_predictions : bool
            Not used

        quantifiers_ : ndarray, shape (n_classes-1, )
            List of quantifiers, one for each model of a FH decomposition. The number is equal to n_classes - 1

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

    def __init__(self, quantifier, estimator_train=None, estimator_test=None, verbose=0):
        super(FrankAndHallQuantifier, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                                     verbose=verbose)
        # attributes
        self.quantifier = quantifier
        self.quantifiers_ = None
        self.classes_ = None

    def fit(self, X, y, predictions_train=None):
        """ Fits all the quanfifiers of a FH decomposition

            First, the method fits the estimators (estimator_train and estimator_test) (if needed) using
            the `fit` method of its superclass

            Then, it creates (using `deepcopy`) the set on quantifiers_ (n_classes-1 quantifiers) and fit them

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, optional, shape (n_examples, n_classes-1) (probs)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train or estimator_test are not instances of OneVsRestClassifier
        """
        if not isinstance(self.quantifier, BaseQuantifier):
            raise ValueError("Class %s: quantifier must be an object of a class derived from quantificationlib.baseQuantifier"
                             % self.__class__.__name__)

        #  binarizing y labels
        label_binarizer = FHLabelBinarizer()
        label_binarizer.fit(y)
        # column i represents y_bin of estimator i (in FH order, from left to right)
        y_bin_fh = label_binarizer.transform(y)

        if isinstance(self.quantifier, WithoutClassifiers):

            self.classes_ = np.unique(y)
            if self.verbose > 0:
                print('Class %s: do not have estimators to fit...' % self.__class__.__name__, end='')

            self.quantifiers_ = np.zeros(len(self.classes_), dtype=object)

            for i in range(len(self.classes_) - 1):
                self.quantifiers_[i] = deepcopy(self.quantifier)
                self.quantifiers_[i].fit(X, y_bin_fh[:, i])

        elif isinstance(self.quantifier, UsingClassifiers):

            if self.estimator_train is not None and not isinstance(self.estimator_train, FrankAndHallClassifier):
                raise ValueError("Class %s: estimator_train must be a FrankAndHallClassifier" % self.__class__.__name__)

            if self.estimator_test is not None and not isinstance(self.estimator_test, FrankAndHallClassifier):
                raise ValueError("Class %s: estimator_test must be a FrankAndHallClassifier" % self.__class__.__name__)

            self.needs_predictions_train = self.quantifier.needs_predictions_train

            super().fit(X, y, predictions_train=predictions_train)

            # Recall that n_estimators==n_classes-1 when using a FrankAndHallClassfier
            self.quantifiers_ = np.zeros(len(self.classes_) - 1, dtype=object)
            for i in range(len(self.classes_) - 1):
                self.quantifiers_[i] = deepcopy(self.quantifier)
                # copy FrankAndHallClassifier's estimators
                if self.estimator_train is not None:
                    self.quantifiers_[i].estimator_train = self.estimator_train.estimators_[i]
                else:
                    self.quantifiers_[i].estimator_train = None
                if self.estimator_test is not None:
                    self.quantifiers_[i].estimator_test = self.estimator_test.estimators_[i]
                else:
                    self.quantifiers_[i].estimator_test = None

                if predictions_train is None:
                    self.quantifiers_[i].fit(X, y_bin_fh[:, i])
                else:
                    preds = np.array([1 - predictions_train[:, i], predictions_train[:, i]]).T
                    self.quantifiers_[i].fit(X, y_bin_fh[:, i], predictions_train=preds)
        return self

    def predict(self, X, predictions_test=None):
        """ Aggregates the prevalences of the quantifiers_ to compute the final prediction

            In this kind of decomposition strategy it is important to ensure that the aggregated consecutive
            prevalencences do not decrease:

            Example:

                Quantifier 1 vs 2-3-4   Prevalence({1}) = 0.3
                Quantifier 1-2 vs 3-4   Prevalence({1,2}) = 0.2
                Quantifier 1-2-3 vs 4   Prevalence({1,2,3}) = 0.6

            This is inconsistent. Following (Destercke, Yang, 2014) the method computes the upper (adjusting from
            left to right) and the lower (from right to left) cumulative prevalences. These sets of values are
            monotonically increasing (from left to right) and monotonically decreasing (from right to left),
            respectively. The average value is assigned to each group and the prevalence for each class is computed as:

                Prevalence({y_k}) = Prevalence({y_1,...,y_k}) - Prevalence({y_1,...,y_k-1})

            Example:

                {1}   {1-2}  {1-2-3}

                0.3   0.3    0.6    Upper cumulative prevalences (adjusting from left to right)

                0.2   0.2    0.6    Lower cumulative prevalences (adjusting from right to left)
                ----------------
                0.25  0.25   0.6    Averaged prevalences

                Prevalence({1}) = 0.25
                Prevalence({2}) = Prevalence({1,2}) - Prevalence({1}) = 0.25 - 0 .25 = 0
                Prevalence({3}) = Prevalence({1,2,3}} - Prevalence({1,2}) = 0.6 - 0.25 = 0.35

                The last class is computed as 1 - the sum of prevalences for the rest of classes

                Prevalence({4}) = 1 - Prevalence({1,2,3}} = 1 - 0.6 = 0.4

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                Predictions for the testing bag

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class

            References
            ----------
            Destercke, S., & Yang, G. (2014, September). Cautious ordinal classification by binary decomposition.
            In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 323-337).
        """
        n_classes = len(self.classes_)

        #  computing the prevalence of the left group (the positive class) of each quantifier
        leftgroup_prevalences = np.zeros(n_classes - 1)  # n_estimators==n_classes-1 when using a FrankAndHallClassfier
        if isinstance(self.quantifier, WithoutClassifiers):
            for i in range(n_classes - 1):
                leftgroup_prevalences[i] = self.quantifiers_[i].predict(X)[1]

        elif isinstance(self.quantifier, UsingClassifiers):
            if predictions_test is None:
                for i in range(n_classes - 1):
                    leftgroup_prevalences[i] = self.quantifiers_[i].predict(X)[1]
            else:
                for i in range(n_classes - 1):
                    preds = np.array([1 - predictions_test[:, i], predictions_test[:, i]]).T
                    leftgroup_prevalences[i] = self.quantifiers_[i].predict(X, predictions_test=preds)[1]

        #  the prevalences are corrected (if needed) to ensure that they increase (not strictly) from left to right
        leftgroup_prevalences = FrankAndHallQuantifier.check_and_correct_prevalences_asc(leftgroup_prevalences)

        # we treat the prevalence for each group following FH decomposition from left to right
        # "c_i" is largest class of the left group: {c_0,c_1,..,c_i}vs{c_i+1,...,c_k}   {left}vs{right}
        prevalences = np.zeros(n_classes)
        # the prevalence of the first class is the same given for the first quantifier (the left group has only one
        # class in this case)
        prevalences[0] = leftgroup_prevalences[0]
        for c_i in range(1, n_classes - 1):
            p_left = leftgroup_prevalences[c_i]
            s = np.sum(prevalences[0:c_i])
            # we use clip() to solve some precision issues, specially  when a prevalence is close to 0.
            # Sometimes p_left - s gives a small negative value
            prevalences[c_i] = np.clip(p_left - s, 0, 1)
        # the prevalence of the last class is 1 minus the sum of the prevalences for the rest of classes
        prevalences[n_classes - 1] = 1 - np.sum(prevalences[0:n_classes - 1])
        return prevalences

    @staticmethod
    def check_and_correct_prevalences_asc(prevalences):
        """ This function checks and corrects the prevalences of a quantifier based on the Frank and Hall decomposition
            that are inconsistent. It is used by FrankAndHallQuantifier.

            To obtain consistent prevalences, we need to ensure that the consecutive probabilities do not decrease.

            Example:

                Quantifier 1 vs 2-3-4   Prevalence({1}) = 0.3
                Quantifier 1-2 vs 3-4   Prevalence({1,2}) = 0.2
                Quantifier 1-2-3 vs 4   Prevalence({1,2,3}) = 0.6

            This is inconsistent. Following (Destercke, Yang, 2014) the method computes the upper (adjusting from
            left to right) and the lower (from right to left) cumulative prevalences. These sets of values are
            monotonically increasing (from left to right) and monotonically decreasing (from right to left),
            respectively. The average value is assigned to each group

            Example:

                {1}   {1-2}  {1-2-3}

                0.3   0.3    0.6    Upper cumulative prevalences (adjusting from left to right)

                0.2   0.2    0.6    Lower cumulative prevalences (adjusting from right to left)
                ----------------
                0.25  0.25   0.6    Averaged prevalences

            Parameters
            ----------
            prevalences : array, shape(n_classes-1, )
                The prevalences of the binary quantifiers of a FrankAndHallQuantifier for a single dataset

            Return
            ------
            prevalences_ok : array, shape(n_classes-1)
                The corrected prevalences ensuring that do not decrease (from left to right)

            References
            ----------
            Sébastien Destercke, Gen Yang. Cautious Ordinal Classification by Binary Decomposition.
            Machine Learning and Knowledge Discovery in Databases - European Conference ECML/PKDD,
            Sep 2014, Nancy, France. pp.323 - 337, 2014,
        """
        ascending = np.all(prevalences[1:] >= prevalences[:-1])
        if ascending:
            return prevalences
        n = len(prevalences)
        # left to right corrections
        prevs1 = np.copy(prevalences)
        for i in range(1, n):
            if prevs1[i] < prevs1[i - 1]:
                prevs1[i] = prevs1[i - 1]
        # right to left correction
        prevs2 = np.copy(prevalences)
        for i in range(n - 1, 0, -1):
            if prevs2[i] < prevs2[i - 1]:
                prevs2[i - 1] = prevs2[i]
        # returning the average of both corrections
        return (prevs1 + prevs2) / 2.0
