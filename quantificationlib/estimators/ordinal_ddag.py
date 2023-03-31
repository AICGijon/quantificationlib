"""
Estimator based on DDAGs (Decision Directed Acyclic Graphs)
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.multiclass import OneVsOneClassifier


class DDAGClassifier(OneVsOneClassifier):
    """ Decision Directed Acyclic Graph ordinal classifier

        This strategy consists on learning a classifier per each pair of classes, thus it requires to fit
        n_classes * (n_classes - 1) / 2 classifiers. For this reason, this class derives from OneVsOneClassifier and
        uses most of its functionalities, mainly to train the binary models.

        However, there are two main differences with respect to OneVsOneClassifier:

        1) This class is used for ordinal classification, it does not make sense to use it for multiclass classification
        2) The rule to make predictions is different. Here, the binary classifiers are arranged into a binary tree in
           which the classifier selected at each node is the one that deals with the two more distant remaining
           classes. Thus, the root contains the classifier that decides between the first class of the order and last
           class, and this idea is recursively applied.

            Example: in a ordinal classification problem with classes ranging from 1-star to 5-star,
            the corresponding DDAG will be

                                                      1|5
                                 1|4                                       2|5
                       1|3                 2|4                   2|4                  3|5
                  1|2       2|3       2|3       3|4         2|3       3|4        3|4       4|5
                1     2   2     3   2     3   3     4     2     3   3     4    3     4   4     5

            Since some subtrees are shared by different branches, for instance, the subtree labeled as node
            2|4 is shared by the right subtree of 1|4 and the left subtree of 2|5, the tree can be depicted in a more
            compact way:
                                                      1|5
                                              1|4             2|5
                                       1|3            2|4            3|5
                                 1|2          2|3             3|4           4|5
                             1          2              3              4             5

            in which all internal nodes (2|4, 2|3 and 3|4) and all leaves, except the first one, and the last one
            (2, 3 and 4) are reached from different paths.


        The class implements two different strategies to compute the probabilities for a given example:

        'full_probabilistic'
            The probabilities computed by each node are propagated through the tree. For those leaves that
            can be reached following different paths (all except the leaves for the first and the last class), the
            probabilities are summed. With this method, all the classes may have a probability greater that 0.

            Example: For a given example, the probability of the left class returned by each model is the following:

            P(1|5) = 0.2
            P(1|4) = 0.1
            P(2|5) = 0.1
            P(1|3) = 0.2
            P(2|4) = 0.3
            P(3|5) = 0.3
            P(1|2) = 0.3
            P(2|3) = 0.4
            P(3|4) = 0.4
            P(4|5) = 0.3

            P(y=1) = P(1|5) * P(1|4) * P(1|3) * P(1|2) =
                     0.2    * 0.1    * 0.2    * 0.3    = 0.0012

            P(y=2) = P(1|5)     * P(1|4)     * P(1|3)     * (1-P(1|2)) +
                     P(1|5)     * P(1|4)     * (1-P(1|3)) * P(2|3)     +
                     P(1|5)     * (1-P(1|4)) * P(2|4)     * P(2|3)     +
                     (1-P(1|5)) * P(2|5)     * P(2|4)     * P(2|3)     =
                     0.2        * 0.1        * 0.2        * 0.7        +
                     0.2        * 0.1        * 0.8        * 0.4        +
                     0.2        * 0.9        * 0.3        * 0.4        +
                     0.8        * 0.1        * 0.3        * 0.4        = 0.0028 + 0.0064 + 0.0216 + 0.0096 = 0.0404

            P(y=3) = P(1|5)     * P(1|4)     * (1-P(1|3)) * (1-P(2|3)) +
                     P(1|5)     * (1-P(1|4)) * P(2|4))    * (1-P(2|3)) +
                     P(1|5)     * (1-P(1|4)) * (1-P(2|4)) * P(3|4)     +
                     (1-P(1|5)) * P(2|5)     * P(2|4))    * (1-P(2|3)) +
                     (1-P(1|5)) * P(2|5)     * (1-P(2|4)) * P(3|4)     +
                     (1-P(1|5)) * (1-P(2|5)) * P(3|5)     * P(3|4)     =
                     0.2        * 0.1        * 0.8        * 0.6        +
                     0.2        * 0.9        * 0.3        * 0.6        +
                     0.2        * 0.9        * 0.7        * 0.4        +
                     0.8        * 0.1        * 0.3        * 0.6        +
                     0.8        * 0.1        * 0.7        * 0.4        +
                     0.8        * 0.9        * 0.3        * 0.4        = 0.0096 + 0.0324 + 0.0504 +
                                                                         0.0144 + 0.0224 + 0.0864 = 0.2156
            P(y=4) = P(1|5)     * (1-P(1|4)) * (1-P(2|4)) * (1-P(3|4)) +
                     (1-P(1|5)) * P(2|5)     * (1-P(2|4)) * (1-P(3|4)) +
                     (1-P(1|5)) * (1-P(2|5)) * P(3|5)     * (1-P(3|4)) +
                     (1-P(1|5)) * (1-P(2|5)) * (1-P(3|5)) * P(4|5)     =
                     0.2        * 0.9        * 0.7        * 0.6        + 0.0756
                     0.8        * 0.1        * 0.7        * 0.6        + 0.0336
                     0.8        * 0.9        * 0.3        * 0.6        + 0.1296
                     0.8        * 0.9        * 0.7        * 0.3        = 0.0756 + 0.0336 + 0.1296 + 0.1512 = 0.3900

            P(y=5) = (1-P(1|5)) * (1-P(2|5)) * (1-P(3|5)) * (1-P(4|5)) =
                     0.8        * 0.9        * 0.7        * 0.7        = 0.3528

            Thus, the probabilities returned by `predict_proba` method would be (0.0012, 0.0404, 02156, 0.3900, 0.3528)

        'winner_node'
            Uses the probabilities of binary estimators to descent until the level previous to the leaves (it is like
            binarizing such probabilities to 0,1). Then, the method returns the probalities of such binary estimator
            for the two consecutive classes involved, and zero for the rest of classes.

            Example: For a given example, the probability of the left class returned by each model is the following:

            P(1|5) = 0.2
            P(1|4) = 0.1
            P(2|5) = 0.1
            P(1|3) = 0.2
            P(2|4) = 0.3
            P(3|5) = 0.3
            P(1|2) = 0.3
            P(2|3) = 0.4
            P(3|4) = 0.4
            P(4|5) = 0.3

            Taking binary decisions from the root, we reach the binary classifier 4|5, thus the returned probabilities
            are (0, 0, 0, 0.3, 0.7)

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

        predict_method: str, optional (default 'full_probabilistic')
            'full_probabilistic'
            The probabilities computed by each node are propagated through the tree. For those leaves that
            can be reached following different paths (all except the leaves for the first and the last class), the
            probabilities are summed. With this method, all the classes may have a probability greater that 0.

            'winner_node'
            Uses the probabilities of binary estimators to descent until the level previous to the leaves (it is like
            binarizing such probabilities to 0,1). Then, the method returns the probalities of such binary estimator
            for the two consecutive classes involved, and zero for the rest of classes.

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        estimator : estimator object
            An estimator object implementing `fit` and `predict_proba`.

        n_jobs : int or None,
            The number of jobs to use for the computation.

        predict_method: str
            The method used by `predict_proba` to compute the class probabilities of a given example

        verbose: int,
            The verbosity level.

        estimators_ : list of ``n_classes * (n_classes - 1) / 2`` estimators
            Estimators used for predictions.

        classes_ : numpy array of shape [n_classes]
            Array containing labels.

        References
        ----------
        José Ramón Quevedo, Elena Montañés, Óscar Luaces, Juan José del Coz: Adapting decision DAGs for multipartite
        ranking. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 115-130).
        Springer, Berlin, Heidelberg.
    """
    def __init__(self, estimator, n_jobs=None, predict_method='full_probabilistic', verbose=0):
        super(DDAGClassifier, self).__init__(estimator=estimator, n_jobs=n_jobs)
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.predict_method = predict_method
        self.verbose = verbose
        self.classes_ = None

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
        return self.classes_[best_classes]  # return best_classes is not correct

    def predict_proba(self, X):
        """ Predict the class probabilities for each example

            Two different methods are implemented depending on the value of `predict_method` attribute

            'full_probabilistic'
            The probabilities computed by each node are propagated through the tree. For those leaves that
            can be reached following different paths (all except the leaves for the first and the last class), the
            probabilities are summed. With this method, all the classes may have a probability greater that 0.

            'winner_node'
            Uses the probabilities of binary estimators to descent until the level previous to the leaves (it is like
            binarizing such probabilities to 0,1). Then, the method returns the probalities of such binary estimator
            for the two consecutive classes involved, and zero for the rest of classes.

            The method uses a recursive auxiliar method to compute the class probabilities

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

        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        probs_ini = np.ones((n_samples, n_classes))
        probs_samples = np.zeros((n_samples, n_classes))
        self.__compute_probabilities(X, probs_samples, probs_accum=probs_ini, left_class=0, right_class=n_classes - 1)
        return probs_samples

    def __compute_probabilities(self, X, probs_samples, probs_accum, left_class, right_class):
        """ Compute the probability for each class applying the binary tree of models

            This method is recursive. It follows the binary tree of models, computing the probability of each class.
            The parameter probs_samples must be a matrix of zeros, shape(n_samples, n_classes). The method uses this
            matrix to compute the final class probabilities by adding the probabilities of the different paths for
            reaching each leaf. The class probabilities of each partial path are stored in probs_accum

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            probs_samples : array, shape(n_examples, n_classes)
                The method computes the probabilities in this matrix. Initially must be set to ones. At the end of the
                recursion proccess this matrix contains the final probabilities

            probs_accum : array, shape(n_examples, n_classes)
                This matrix contains the accumulated probabilities from the root to the given node defined, by
                the parameters first and last

            left_class :  int, initially must be 0
                The index of the left class of the binary model at the current node

            right_class : int, initially must be equal to n_classes-1
                The index of the right class of the binary model at the current node
        """
        if left_class == right_class:
            # it is a leaf, we have to add the probabilities of this branch (are stored in probs_accum
            probs_samples += probs_accum
        else:
            # internal node
            n_classes = len(self.classes_)

            pos_estim = sum(list(range(n_classes - left_class, n_classes))) + (right_class - left_class - 1)
            predictions = self.estimators_[pos_estim].predict_proba(X)

            n_samples = len(predictions)

            predictions_left = predictions[:, 0]
            predictions_right = predictions[:, 1]

            if self.predict_method == 'winner_node' and right_class - left_class != 1:
                # binarizing the probabilities for the prediction method 'winner_node'. This is done in all cases
                # except for the model previous to the leaves (when right_class-left_class == 1)
                predictions_left[predictions_left < 0.5] = 0
                predictions_left[predictions_left >= 0.5] = 1
                predictions_right[predictions_right <= 0.5] = 0
                predictions_right[predictions_right > 0.5] = 1

            probs_left_accum = np.zeros((n_samples, n_classes))
            probs_right_accum = np.zeros((n_samples, n_classes))

            # left side classes
            for cl in range(left_class, right_class):
                probs_left_accum[:, cl] = probs_accum[:, cl] * predictions_left

            # right side classes
            for cl in range(left_class + 1, right_class + 1):
                probs_right_accum[:, cl] = probs_accum[:, cl] * predictions_right

            # process the probabilities of each subtree
            self.__compute_probabilities(X, probs_samples, probs_left_accum, left_class, right_class - 1)
            self.__compute_probabilities(X, probs_samples, probs_right_accum, left_class + 1, right_class)

