"""
Estimators based on Frank and Hall decomposition
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
from sklearn.preprocessing import LabelBinarizer

from copy import copy, deepcopy
from joblib import Parallel, delayed

from quantificationlib.metrics.binary import binary_kld


class FrankAndHallClassifier(BaseEstimator, ClassifierMixin):
    """ Ordinal Classifier following Frank and Hall binary decomposition

        This type of decomposition works as follows. For instance, in a ordinal classification problem with classes
        ranging from 1-star to 5-star, Frank and Hall (FH) decompositon trains 4 binary classifiers:
        1 vs 2-3-4-5, 1-2 vs 3-4-5, 1-2-3 vs 4-5, 1-2-3-4 vs 5 and combines their predictions.

        Parameters
        ----------
        estimator : estimator object (default=None)
            An estimator object implementing `fit` and one of `predict` or `predict_proba`. It is the base estimator
            used to learn the set of binary classifiers

        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        params_fit : list of dictionaries with parameters for each binary estimator, optional
            Example: 5 classes/4 binary estimators:

                params_fit = [{'C':0.0001} , {'C':0.000001}, {'C':0.000001}, {'C':0.01}]

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        estimator : estimator object
            The base estimator used to build the FH decomposition

        n_jobs : int or None,
            The number of jobs to use for the computation.

        params_fit : list of dictionaries
             It has the parameters for each binary estimator

        verbose : int
            The verbosity level. The default value, zero, means silent mode

        classes_ : ndarray, shape (n_classes, )
            Class labels

        estimators_ : ndarray, shape(n_classes-1,)
            List of binary estimators following the same order of the Frank and Hall decomposition:
                estimators_[0] -> 1 vs 2-3-4-5
                estimators_[1] -> 1-2 vs 3-4-5
                ...

        label_binarizer_ :  FHLabelBinarizer object
            Object used to transform multiclass labels to binary labels and vice-versa

        References
        ----------
        Eibe Frank and Mark Hall. 2001. A simple approach to ordinal classification.
        In Proceedings of the European Conference on Machine Learning. Springer, 145156.
    """

    def __init__(self, estimator=None, n_jobs=None, verbose=0, params_fit=None):
        self.estimator = estimator
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.params_fit = params_fit
        # computed variables
        self.classes_ = None
        self.estimators_ = None
        self.label_binarizer_ = None

    def _fit_binary(self, X, y_bin, pos_class):
        """ Fits just one binary classifier of the FH decomposition

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            y_bin : (sparse) array-like, shape (n_examples, )
                True classes of the binary problem. The label of classes are: 1 (positive class) and 0 (negative class)

            pos_class : int
                Index of the estimator in the FH decomposition, beginning in 0. pos_class + 1 is also the number of the
                last class that belongs to the left group
        """
        if self.verbose:
            print("Fitting estimator [1..{}] vs [{}..{}]".format(pos_class + 1, pos_class + 2, len(self.classes_)))
        clf = deepcopy(self.estimator)
        ######
        if self.params_fit is not None:
            params = self.params_fit[pos_class]
            clf.set_params(**params)
            print("Estimator ", pos_class, " params: ", params)
        #######
        clf.fit(X, y_bin)
        return clf

    def fit(self, X, y):
        """ Fits the set of estimators for the training set following the Frank and Hall decomposition

            It learns a list of binary estimators following the same order of the Frank and Hall decomposition:
                estimators_[0] -> 1 vs 2-3-4-5
                estimators_[1] -> 1-2 vs 3-4-5
                ...

            The left group of each classifier ({1}, {1,2}, ...) is the positive class

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            y : (sparse) array-like, shape (n_examples, )
                True classes

            Raises
            ------
            ValueError
                When estimator is None
        """
        if self.estimator is None:
            raise ValueError("An estimator is needed for %s objects", self.__class__.__name__)

        # if self.estimators_ is not None:
        # self.estimators_ = None

        self.label_binarizer_ = FHLabelBinarizer()
        y_bin_fh = self.label_binarizer_.fit_transform(y)  # column i contains y_bin for estimator i (in FH order)
        self.classes_ = self.label_binarizer_.classes_
        n_classes = len(self.classes_)

        X, y = check_X_y(X, y, accept_sparse=True)

        # fit the estimator for each binary combination of classes, n_estimators = n_classes -1
        self.estimators_ = np.empty((n_classes - 1,), dtype=object)

        # In cases where individual estimators are very fast to train, setting n_jobs > 1 can result in slower
        # performance due to the overhead of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_binary)(X, y_bin_fh[:, pos_class], pos_class)
                                                        for pos_class in range(n_classes - 1))
        return self

    def predict(self, X):
        """ Predict the class for each testing example

            The method computes the probability of each class (using `predict_proba`) and returns the class with
            highest probability

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            Returns
            -------
            An array, shape(n_examples, ) with the predicted class for each example

            Raises
            ------
            NotFittedError
                When the estimators are not fitted yet
        """
        if self.classes_ is None:
            raise NotFittedError("This instance of %s class is not fitted yet", type(self).__name__)

        probs_samples = self.predict_proba(X)
        best_classes = np.argmax(probs_samples, axis=1)
        return self.classes_[best_classes]  # this returns the label of the winner class instead of its index

    def predict_proba(self, X):
        """ Predict the class probabilities for each example following the original rule proposed by Frank & Hall

            If the classes are c_1 to c_k:

                Pr(y = c_1) = Pr (y <= c_1)
                Pr(y = c_i) = Pr(y > c_i−1)  x (1 − Pr(y > c_i)) ; 1 < i < k
                Pr(y = c_k) = Pr(y > c_k−1)

                Notice that :  :math:`sum_{i=1}^{i=k} Pr(c_i) \neq 1`

            Example with 5 classes

                We have 4 binary estimators that return two probabilities: the probability of the left group and the
                probability of the right group, denoted as e_i.left and e_i.right respectively, in which i is the
                number of the estimator 1<=i<k

                Estimator 0:	c1  |   c2, c3, c4, c5          e1.left	| e1.right
                Estimator 2:	c1, c2  |   c3, c4, c5          e2.left	| e2.right
                Estimator 3:	c1, c2, c3  |   c4, c5          e3.left	| e3.right
                Estimator 4:	c1, c2, c3  c4  |   c5          e4.left	| e4.right

                Pr(y = c_1) = e1.left
                Pr(y = c_2) = e1.right x e2.left
                Pr(y = c_3) = e2.right x e3.left
                Pr(y = c_4) = e3.right x e4.left
                Pr(y = c_5) = e4.right


            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            Returns
            -------
            An array, shape(n_examples, n_classes) with the class probabilities for each example

            Raises
            ------
            NotFittedError
                When the estimators are not fitted yet
        """
        if self.classes_ is None:
            raise NotFittedError("This instance of %s class is not fitted yet", type(self).__name__)

        # computing the probabilites of the left group for all estimators
        # notice that the probabilities of the right group is 1 - probability of the left group
        predictions = self.__compute_binary_proba(X)

        n_classes = len(self.classes_)
        n_samples = len(predictions)
        probs_samples = np.zeros((n_samples, n_classes))
        # the probability of the first class is just the probability of the left group of the first estimator
        probs_samples[:, 0] = predictions[:, 0]
        # for the rest of class (except the last one), the probability of that class is the product of the probability
        # of the right group of the c_i-1 estimator, and the probability of the left group of the c_i estimator
        for c_i in range(1, n_classes - 1):
            # Pr(y = c_i) = p_right_estimator_i-1 * p_left_estimator_i
            probs_samples[:, c_i] = 1 - predictions[:, c_i - 1] * predictions[:, c_i]
        # the probability of the last class is the probability of the right group of the last estimator
        probs_samples[:, n_classes - 1] = 1 - predictions[:, n_classes - 1 - 1]

        return probs_samples

    def __compute_binary_proba(self, X):
        """ Compute the class probabilities of the internal binary estimators

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)

            Returns
            -------
            ndarray, shape(n_samples, n_estimators)
                For each sample, this matrix contains the probabily that such sample belongs to left group of classes
                of each estimator of the FH decomposition
                Recall that n_estimators = n_classes-1
        """

        if self.classes_ is None:
            raise NotFittedError("This instance of %s class is not fitted yet", type(self).__name__)

        n_estimators = len(self.classes_) - 1
        # we compute the predictions for the first estimator to know the lenght of the predictions. Sometimes,
        # for instance when a CV_estimator is used, len(X) does not match with len(predictions)
        predictions = self.estimators_[0].predict_proba(X)
        # the method just returns the probabilities of the left_group
        predictions_left = np.zeros((len(predictions), n_estimators))
        predictions_left[:, 0] = predictions[:, 1]
        # now for the rest of estimators
        for i in range(1, n_estimators):
            predictions = self.estimators_[i].predict_proba(X)
            predictions_left[:, i] = predictions[:, 1]

        return predictions_left


class FrankAndHallMonotoneClassifier(FrankAndHallClassifier):
    """ Ordinal Classifier following Frank and Hall binary decomposition but returning consistent probabilities

        This type of decomposition works as follows. For instance, in a ordinal classification problem with classes
        ranging from 1-star to 5-star, Frank and Hall (FH) decompositon trains 4 binary classifiers:
        1 vs 2-3-4-5, 1-2 vs 3-4-5, 1-2-3 vs 4-5, 1-2-3-4 vs 5 and combines their predictions.

        The difference with FrankAndHallClassifier is that the original method devised by Frank & Hall was intented
        just for crips predictions. The computed probabilities for all classes may be not consistent (their sum is
        not 1 in many cases)

        Following (Destercke, Yang, 2014) this class computes the upper (adjusting from  left to right) and the lower
        (from right to left) cumulative probabilities for each group of classes. These sets of values are
        monotonically increasing (from left to right) and monotonically decreasing (from right to left), respectively.
        The final probability assigned to each group is the average of both values, and the probality of each class
        is computed as:

                Pr({y_k}) = Pr({y_1,...,y_k}) - Pr({y_1,...,y_k-1})

        Parameters
        ----------
        estimator : estimator object (default=None)
            An estimator object implementing `fit` and one of `predict` or `predict_proba`. It is the base estimator
            used to learn the set of binary classifiers

        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        params_fit : list of dictionaries with parameters for each binary estimator, optional
            Example: 5 classes/4 binary estimators:

                params_fit = [{'C':0.0001} , {'C':0.000001}, {'C':0.000001}, {'C':0.01}]

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        estimator : estimator object
            The base estimator used to build the FH decomposition

        n_jobs : int or None,
            The number of jobs to use for the computation.

        verbose : int
            The verbosity level. The default value, zero, means silent mode

        params_fit : list of dictionaries
             It has the parameters for each binary estimator (not used in this class)

        classes_ : ndarray, shape (n_classes, )
            Class labels

        estimators_ : ndarray, shape(n_classes-1,)
            List of binary estimators following the same order of the Frank and Hall decomposition:
                estimators_[0] -> 1 vs 2-3-4-5
                estimators_[1] -> 1-2 vs 3-4-5
                ...

        label_binarizer_ :  FHLabelBinarizer object
            Object used to transform multiclass labels to binary labels and vice-versa

        References
        ----------
        Sébastien Destercke, Gen Yang. Cautious Ordinal Classification by Binary Decomposition.
        Machine Learning and Knowledge Discovery in Databases - European Conference ECML/PKDD,
        Sep 2014, Nancy, France. pp.323 - 337, 2014,
    """
    def __init__(self, estimator=None, n_jobs=None, verbose=0, params_fit=None):
        super(FrankAndHallMonotoneClassifier, self).__init__(estimator=estimator, n_jobs=n_jobs, verbose=verbose,
                                                             params_fit=params_fit)

    def fit(self, X, y):
        """ Fits the set of estimators for the training set following the Frank and Hall decomposition

            It learns a list of binary estimators following the same order of the Frank and Hall decomposition:
                estimators_[0] -> 1 vs 2-3-4-5
                estimators_[1] -> 1-2 vs 3-4-5
                ...

            The left group of each classifier ({1}, {1,2}, ...) is the positive class

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            y : (sparse) array-like, shape (n_examples, )
                True classes

            Raises
            ------
            ValueError
                When estimator is None
        """
        super().fit(X, y)
        return self

    def predict(self, X):
        """ Predict the class for each testing example

            The method computes the probability of each class (using `predict_proba`) and returns the class with
            highest probability

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            Returns
            -------
            An array, shape(n_examples, ) with the predicted class for each example

            Raises
            ------
            NotFittedError
                When the estimators are not fitted yet
        """
        probs_samples = self.predict_proba(X)
        best_classes = np.argmax(probs_samples, axis=1)
        return self.classes_[best_classes]

    def predict_proba(self, X):
        """ Predict the class probabilities for each example following a new rule (different from the original
            one proposed by Frank & Hall)

            To obtain consistent probabilities, we need to ensure that the aggregated consecutive
            probabilities do not decrease.

            Example:

                Classifier 1 vs 2-3-4   Pr({1}) = 0.3
                Classifier 1-2 vs 3-4   Pr({1,2}) = 0.2
                Classifier 1-2-3 vs 4   Pr({1,2,3}) = 0.6

            This is inconsistent. Following (Destercke and Yang, 2014) the method computes the upper (adjusting from
            left to right) and the lower (from right to left) cumulative probabilities. These sets of values are
            monotonically increasing (from left to right) and monotonically decreasing (from right to left),
            respectively. The average value is assigned to each group and the probability for each class is computed as:

                Pr({y_k}) = Pr({y_1,...,y_k}) - Pr({y_1,...,y_k-1})

            Example:

                {1}   {1-2}  {1-2-3}

                0.3   0.3    0.6    Upper cumulative probabilities (adjusting from left to right)

                0.2   0.2    0.6    Lower cumulative probabilities (adjusting from right to left)
                ----------------
                0.25  0.25   0.6    Averaged probability

                Pr({1}) = 0.25
                Pr({2}) = Pr({1,2}) - Pr({1}) = 0.25 - 0 .25 = 0
                Pr({3}) = Pr({1,2,3}} - Pr({1,2}) = 0.6 - 0.25 = 0.35

                The last class is computed as 1 - the sum of the probabilities for the rest of classes

                Pr({4}) = 1 - Pr({1,2,3}} = 1 - 0.6 = 0.4

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            Returns
            -------
            An array, shape(n_examples, n_classes) with the class probabilities for each example

            Raises
            ------
            NotFittedError
                When the estimators are not fitted yet
        """
        if self.classes_ is None:
            raise NotFittedError("This instance of %s class is not fitted yet", type(self).__name__)

        predictions_left = self._FrankAndHallClassifier__compute_binary_proba(X)

        n_classes = len(self.classes_)
        n_samples = len(predictions_left)

        # the probabilities are corrected (if needed) to ensure that they increase (not strictly) from left to right
        predictions_left = FrankAndHallMonotoneClassifier.__check_and_correct_probabilities_asc(predictions_left)

        # we treat the prevalence for each group following FH decomposition from left to right
        # "c_i" is largest class of the left group: {c_0,c_1,..,c_i}vs{c_i+1,...,c_k}   {left}vs{right}
        probs_samples = np.zeros((n_samples, n_classes))
        # the probability of the first class is the same given for the first estimator (the left group has only one
        # class in this case)
        probs_samples[:, 0] = predictions_left[:, 0]
        for c_i in range(1, n_classes - 1):
            p_left = predictions_left[:, c_i]
            s = np.sum(probs_samples[:, 0:c_i], axis=1)
            # we use clip() to solve some precision issues, specially  when a probability is close to 0.
            # Sometimes p_left - s gives a small negative value
            probs_samples[:, c_i] = np.clip(p_left - s, 0, 1)
        # the probability of the last class is 1 minus the sum of the prevalences for the rest of classes
        probs_samples[:, n_classes - 1] = 1 - np.sum(probs_samples[:, 0:n_classes - 1], axis=1)
        return probs_samples

    @staticmethod
    def __check_and_correct_probabilities_asc(probabilities):
        """ This function checks and corrects those probabilities of the binary models of a Frank and Hall estimator
            that are inconsistent. It is used by FrankAndHallMonotoneClassifier.

            To obtain consistent probabilities, we need to ensure that the consecutive probabilities do not decrease.

            Example:

                Classifier 1 vs 2-3-4   Pr({1}) = 0.3
                Classifier 1-2 vs 3-4   Pr({1,2}) = 0.2
                Classifier 1-2-3 vs 4   Pr({1,2,3}) = 0.6

            This is inconsistent. Following (Destercke and Yang, 2014) the method computes the upper (adjusting from
            left to right) and the lower (from right to left) cumulative probabilities. These sets of values are
            monotonically increasing (from left to right) and monotonically decreasing (from right to left),
            respectively. The average value is assigned to each group.

            Example:

                {1}   {1-2}  {1-2-3}

                0.3   0.3    0.6    Upper cumulative probabilities (adjusting from left to right)

                0.2   0.2    0.6    Lower cumulative probabilities (adjusting from right to left)
                ----------------
                0.25  0.25   0.6    Averaged probability

            Parameters
            ----------
            probabilities : array, shape(n_examples, n_classes-1)
                The probabilities of the binary models of a FrankAndHonotone estimator for a complete dataset

            Return
            ------
            probabilities_ok : array, shape(n_examples, n_classes-1)
                The corrected probabilities ensuring that do not decrease (from left to right)

            References
            ----------
            Sébastien Destercke, Gen Yang. Cautious Ordinal Classification by Binary Decomposition.
            Machine Learning and Knowledge Discovery in Databases - European Conference ECML/PKDD,
            Sep 2014, Nancy, France. pp.323 - 337, 2014,
        """
        ascending = np.all(probabilities[:, 1:] >= probabilities[:, :-1], axis=1)
        samples_to_correct = np.nonzero(ascending is False)[0]
        if len(samples_to_correct) == 0:
            return probabilities
        else:
            probabilities_ok = np.copy(probabilities)
        # correct those samples in which their probabilities are incosistent
        for sample in samples_to_correct:
            n = len(probabilities_ok[sample])
            # left to right corrections
            probs1 = np.copy(probabilities_ok[sample])
            for i in range(1, n):
                if probs1[i] < probs1[i - 1]:
                    probs1[i] = probs1[i - 1]
            # right to left correction
            probs2 = np.copy(probabilities_ok[sample])
            for i in range(n - 1, 0, -1):
                if probs2[i] < probs2[i - 1]:
                    probs2[i - 1] = probs2[i]
            # storing the average of both corrections
            probabilities_ok[sample] = (probs1 + probs2) / 2.0
        return probabilities_ok


class FrankAndHallTreeClassifier(FrankAndHallClassifier):
    """ Ordinal Classifier following Frank and Hall binary decomposition but organizing the binary models in a
        tree to compute the predictions

        This type of decomposition works as follows. For instance, in a ordinal classification problem with classes
        ranging from 1-star to 5-star, Frank and Hall (FH) decompositon trains 4 binary classifiers:
        1 vs 2-3-4-5, 1-2 vs 3-4-5, 1-2-3 vs 4-5, 1-2-3-4 vs 5 and combines their predictions.

        The difference with FrankAndHallClassifier is that the original method devised by Frank & Hall computes the
        probability of each class applying the binary models from left to right: 1 vs 2-3-4-5, 1-2 vs 3-4-5, and so on.
        This classifier is based on the method proposed by (San Martino, Gao and Sebastiani, 2016). The idea is to
        build a binary tree with the binary models of the Frank and Hall decomposition, selecting at each point of the
        tree the best possible model according to their quantification performance (applying PCC algorithm with each
        binary classifier and using the KLD as performance measure).

        Example:
                                             1-2-3 vs 4-5
                         1 vs 2-3-4-5                            1-2-3-4 vs 5
                      1                1-2 vs 3-4-5           4                5
                                     2              3

        Parameters
        ----------
        estimator : estimator object (default=None)
            An estimator object implementing `fit` and one of `predict` or `predict_proba`. It is the base estimator
            used to learn the set of binary classifiers

        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        performance_measure : a binary quantification performance measure, (default=binary_kld)
            The binary quantification performance measure used to estimate the goodness of each binary classifier used
            as quantifier

        params_fit : list of dictionaries with parameters for each binary estimator, optional
            Example: 5 classes/4 binary estimators:

                params_fit = [{'C':0.0001} , {'C':0.000001}, {'C':0.000001}, {'C':0.01}]

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        estimator : estimator object
            The base estimator used to build the FH decomposition

        n_jobs : int or None,
            The number of jobs to use for the computation.

        performance_measure : str, or any binary quantification performance measure
            The binary quantification performance measure used to estimate the goodness of each binary classifier used
            as quantifier

        verbose : int
            The verbosity level. The default value, zero, means silent mode

        params_fit : list of dictionaries
             It has the parameters for each binary estimator

        classes_ : ndarray, shape (n_classes, )
            Class labels

        estimators_ : ndarray, shape(n_classes-1,)
            List of binary estimators following the same order of the Frank and Hall decomposition:
                estimators_[0] -> 1 vs 2-3-4-5
                estimators_[1] -> 1-2 vs 3-4-5
                ...

        label_binarizer_ :  FHLabelBinarizer object
            Object used to transform multiclass labels to binary labels and vice-versa

        tree_ : A tree
            A tree with the binary classifiers ordered by their quantification performance (using KLD or other measure)

        References
        ----------
        Giovanni Da San Martino, Wei Gao, and Fabrizio Sebastiani. 2016a. Ordinal text quantification.
        In Proceedings of the International ACM SIGIR Conference on  Research and Development
        in Information Retrieval. 937940.

        Giovanni Da San Martino,Wei Gao, and Fabrizio Sebastiani. 2016b.
        QCRI at SemEval-2016 Task 4: Probabilistic methods for binary and ordinal quantification.
        In Proceedings of the 10th InternationalWorkshop on Semantic Evaluation (SemEval’16).
        Association for Computational Linguistics, A, 5863.
    """
    def __init__(self, estimator=None, n_jobs=None, verbose=0, performance_measure=binary_kld, params_fit=None):
        super(FrankAndHallTreeClassifier, self).__init__(estimator=estimator, n_jobs=n_jobs, verbose=verbose,
                                                         params_fit=params_fit)
        # specific attributes
        self.performance_measure = performance_measure
        self.tree_ = None

    def fit(self, X, y):
        """ Fits the set of estimators for the training set following the Frank and Hall decomposition and builds
            the binary tree to organize such estimators

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            y : (sparse) array-like, shape (n_examples, )
                True classes

            Raises
            ------
            ValueError
                When estimator is None
        """
        super().fit(X, y)

        # building the tree
        self.__generate_tree(X, y)
        if self.verbose > 0:
            print(self.tree_)
        return self

    def predict(self, X):
        """ Predict the class for each testing example

            The method computes the probability of each class (using `predict_proba`) and returns the class with
            highest probability

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            Returns
            -------
            An array, shape(n_examples, ) with the predicted class for each example

            Raises
            ------
            NotFittedError
                When the estimators are not fitted yet
        """
        probs_samples = self.predict_proba(X)
        best_classes = np.argmax(probs_samples, axis=1)
        return self.classes_[best_classes]

    def predict_proba(self, X):
        """ Predict the class probabilities for each example applying the binary tree of models

            Example:
                                             1-2-3 vs 4-5
                         1 vs 2-3-4-5                            1-2-3-4 vs 5
                      1                1-2 vs 3-4-5           4                5
                                     2              3

                 Imagine that for a given example the probabily returned by each model are the following (the models
                 return the probability of the left group of classes):

                 Pr({1,2,3}) = 0.2
                 Pr({1}) = 0.9
                 Pr({1,2,3,4}) = 0.7
                 Pr({1,2}) = 0.4

                 with tha values, the probability for each class will be:

                 Pr({1}) = Pr({1,2,3}) * Pr({1}) = 0.2 * 0.9 = 0.18
                 Pr({2}) = Pr({1,2,3}) * (1-Pr({1})) * Pr({1,2}) = 0.2 * 0.1 * 0.4 = 0.008
                 Pr({3}) = Pr({1,2,3}) * (1-Pr({1})) * (1-Pr({1,2})) = 0.2 * 0.1 * 0.6 = 0.012
                 Pr({4}) = (1-Pr({1,2,3}) * Pr{1,2,3,4}) = 0.8 * 0.7 = 0.56
                 Pr({5}) = (1-Pr({1,2,3}) * (1-Pr{1,2,3,4})) = 0.8 * 0.3 = 0.24

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            Returns
            -------
            An array, shape(n_examples, n_classes) with the class probabilities for each example

            Raises
            ------
            NotFittedError
                When the estimators are not fitted yet
        """
        if self.classes_ is None:
            raise NotFittedError("This instance of %s class is not fitted yet", type(self).__name__)

        if self.tree_ is None:
            raise ValueError('Tree does not exist')

        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        probs_samples = np.ones((n_samples, n_classes))
        self.__compute_probabilities(self.tree_, X, probs_samples, classes_in_subtree=[])

        return probs_samples

    def __generate_tree(self, X, y):
        """ Build the tree for a given dataset X,y. This dataset is used to measure the quantification performance
            of each binary classifier.

            The method first computes the quantification performance of each binary classifier and then recursively
            builds the binary tree of models using an auxiliar recursive function

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            y : (sparse) array-like, shape (n_examples, )
                True classes

            Raises
            ------
            ValueError
                When estimator is None
        """
        # obtain the probabilistic predictions of each binary classifier (needed to apply PCC). Recall that these
        # models return the probabilities of the classes of the left group
        predictions_left = self._FrankAndHallClassifier__compute_binary_proba(X)

        # compute the predicted prevalence of the left group (PCC algorithm, it is just the mean)
        p_pred = np.mean(predictions_left, axis=0)

        # computing the quantification performance of each binary classifier applying the selected
        # performance measure
        n_samples = len(y)
        n_estimators = len(self.classes_) - 1
        scores = np.zeros(n_estimators)
        for i in range(n_estimators):
            # compute the true prevalence of the left group for such estimator
            y_val_bin = self.label_binarizer_.transform(y)[:, i]
            num_left = np.count_nonzero(y_val_bin == 0)  # left 0, right +1
            p_true = num_left / float(n_samples)
            scores[i] = self.performance_measure(p_true, p_pred[i])
            if self.verbose:
                print("Model", i, "Quantification performance: ", scores[i])

        # build the tree using these scores
        self.tree_ = self.__build_recursive_tree(np.asarray(scores), 0, n_estimators - 1)
        return self

    def __build_recursive_tree(self, scores, first, last):
        """ Builds recursively the binary tree

            Parameters
            ----------
            scores : array, shape(n_estimators,)
                Array with the quantification performance score (default, kld) of each binary classifier

            first :  int,
                The index of the first class of the range of classes that belong to that subtree

            last : int,
                The index of the last class of the range of classes that belong to that subtree
        """
        if first > last:
            return QTree()
        else:
            # Choose best estimator (lower score) between first and last
            pos_best = first + np.argmin(scores[first:last + 1])  # same order scores and estimators

            left_subtree = self.__build_recursive_tree(scores, first, pos_best - 1)
            right_subtree = self.__build_recursive_tree(scores, pos_best + 1, last)

            return QTree(self, pos_best, left_subtree, right_subtree)

    def __compute_probabilities(self, node, X, probs_samples, classes_in_subtree):
        """ Compute the probability for each class applying the binary trees of models

            This method is recursive. It follows the binary tree of models, computing the probability of each class.
            The parameter probs_samples must be a matrix of ones, shape(n_samples, n_classes). The method uses this
            matrix to compute the probabilities by multiplying the probabilities of the corresponding models for
            each pair example/class, depending on the structure of the tree

            Parameters
            ----------
            node : a tree
                Initial this value is equal to the root of the tree

            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            probs_samples : array, shape(n_examples, n_classes)
                The method computes the probabilities in this matrix. Initially must be set to ones. At the end of the
                recursion proccess this matrix contains the final probabilities

            classes_in_subtree: array
                Labels of all the classes that have a leaf on the tree whose root is node
        """
        if not node.is_leaf():
            current_estimator = self.estimators_[node.pos_estimator]
            classes_left = self.classes_[:node.pos_estimator + 1]
            classes_right = self.classes_[node.pos_estimator + 1:]

            if hasattr(current_estimator, "predict_proba"):
                predictions = current_estimator.predict_proba(X)
                predictions_left = predictions[:, 1]
                predictions_right = predictions[:, 0]
            else:
                raise ValueError("Need predict_proba in current_estimator with objects of class",
                                 self.__class__.__name__)

            if len(classes_in_subtree) == 0:  # first call
                classes_left_subtree = classes_left
                classes_right_subtree = classes_right
            else:
                classes_left_subtree = np.setdiff1d(classes_in_subtree, classes_right)
                classes_right_subtree = np.setdiff1d(classes_in_subtree, classes_left)

            for cl in classes_left_subtree:
                pos_cl = self.classes_.tolist().index(cl)
                probs_samples[:, pos_cl] *= predictions_left

            for cl in classes_right_subtree:
                pos_cl = self.classes_.tolist().index(cl)
                probs_samples[:, pos_cl] *= predictions_right

            self.__compute_probabilities(node.left, X, probs_samples, classes_left_subtree)
            self.__compute_probabilities(node.right, X, probs_samples, classes_right_subtree)


class QTree:
    """ Auxiliar class to represent the binary trees needed by FrankAndHallTreeClassifier

        Parameters
        ----------
        fhqtree: FrankAndHallTreeClassifier object  (default=None)

        pos_estimator : int, (default=0)
            Index of the estimator in the order defined by the Frank and Hall decomposition: 1 vs 2-3-4-5, 1-2 vs 3-4-5
            and so on.

        left: a QTree object (default=None)
            Left subTree of this node


        right: a QTree object (default=None)
            Right subTree of this node
    """

    def __init__(self, fhtree=None, pos_estimator=0, left=None, right=None):
        self.fhtree = fhtree
        self.pos_estimator = pos_estimator
        self.left = left
        self.right = right

    def is_leaf(self):
        """ Check whether it is a leaf or not """
        return self.left is None and self.right is None

    def __str__(self):
        """ Generates a str representation of the object
            It is based on a recursive function: __rec_str
        """
        level = 0
        # s = self.__rec_str(level)
        s = self.__rec_str(level, classes_father=[], classes_visited=[])
        return s

    def __rec_str(self, level, classes_father, classes_visited):
        prefix = "\t" * level
        if self.is_leaf():
            s = prefix + str(classes_father[0])
            classes_visited.append(classes_father[0])
        else:
            # classes 0..pos_class  vs the rest
            classes_left = self.fhtree.classes_[:self.pos_estimator + 1]
            classes_right = self.fhtree.classes_[self.pos_estimator + 1:]

            s = prefix + str(classes_left.tolist()) + "vs" + str(classes_right.tolist())

            classes_left_pending = np.setdiff1d(classes_left, classes_visited)
            classes_right_pending = np.setdiff1d(classes_right, classes_visited)

            s_left = self.left.__rec_str(level + 1, classes_left_pending, classes_visited)
            s_right = self.right.__rec_str(level + 1, classes_right_pending, classes_visited)

            s = s + "\n" + s_left + "\n" + s_right
        return s


class FHLabelBinarizer(LabelBinarizer):
    """ Binarize labels in a Frank and Hall decomposition

        This type of decomposition works as follows. For instance, in a ordinal classification problem with classes
        ranging from 1-star to 5-star, Frank and Hall (FH) decompositon trains 4 binary classifiers:
        1 vs 2-3-4-5, 1-2 vs 3-4-5, 1-2-3 vs 4-5, 1-2-3-4 vs 5 and combines their predictions.

        To train all these binary classifiers, one needs to convert the original ordinal labels to binary labels
        for each of the binary problems of the Frank and Hall decomposition. FHLabelBinarizer makes this process
        easy using the transform method.

         Parameters
         ----------
         neg_label : int (default: 0)
             Value with which negative labels must be encoded.

         pos_label : int (default: 1)
             Value with which positive labels must be encoded.

         sparse_output : boolean (default: False)
             True if the returned array from transform is desired to be in sparse CSR format.
        """
    def __init__(self, neg_label=0, pos_label=1):
        super(FHLabelBinarizer, self).__init__(neg_label=neg_label, pos_label=pos_label, sparse_output=False)

    def transform(self, y):
        """ Transform ordinal labels to the Frank and Hall binary labels

            Parameters
            ----------
            y : array, (n_samples,)
                Class labels for a set of examples

            Returns
            -------
            y_bin_fh : array, (n_samples, n_classes)
                Each column contains the binary labels for the consecutive binary problems of a Frank and Hall
                decomposition from left to right. For instance, in a 4-class problem, each column corresponds to
                the following problems:

                1st column: 1 vs 2-3-4
                2nd column: 1-2 vs 3-4
                3rd column: 1-2-3 vs 4
                4ht column: (not really used)
        """
        y_bin = super().transform(y)
        y_bin_fh = copy(y_bin)
        for i in range(len(self.classes_)):
            y_bin_fh[:, i] = np.sum(y_bin[:, 0:i + 1], axis=1)
        return y_bin_fh
