"""
Multiclass versions for quantifiers based on the Energy Distance
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from sklearn.utils import check_X_y, check_array
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.stats import rankdata

from quantificationlib.base import UsingClassifiers, WithoutClassifiers
from quantificationlib.optimization import compute_ed_param_train, compute_ed_param_test, solve_ed


class EDy(UsingClassifiers):
    """ Multiclass EDy method

        As described in (Castaño et al 2019), the predicted prevalences can be analytically calculated solving an
        optimization problem (with quadprog.solve_qp in this library). All ED-based methods share several functions
        in distribution_matching.utils. These functions are used to compute the elements of the optimization
        problem (`compute_ed_param_train`, `compute_ed_param_test`) and to solve the optimization problem (`solve_ed`)

        This class (as every other class based on distribution matching using classifiers) works in two different ways:

        1) Two estimators are used to classify training examples and testing examples in order to
           compute the distribution of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the fit/predict methods. This is useful
           for synthetic/artificial experiments

        The idea in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC and AC/PAC). In the first case, estimators are only trained once and can
        be shared for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the distribution of each class individually

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

        distance : distance function (default=manhattan_distances)
            It is the function used to compute the distance between every pair of examples

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimator_train and estimator_test could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because EDy quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
             This means that predictions_train_/predictions_test_ contain probabilistic predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit` method whenever the true labels of the training set are needed, instead of y

        train_n_cls_i_ : ndarray, shape(n_classes, 1)
            Number of the examples of each class in the training set. Used to compute average distances

        train_distrib_ : Dict, the keys are the labels of the classes (classes_)
            Each key has associated a ndarray with the predictions, shape (train_n_cls_i_[i], 1) (binary quantification
            problems) or (train_n_cls_i_[i], n_classes) (multiclass quantification problems)

        K_ : ndarray, shape (n_classes, n_classes)
            Average distance between the examples in the training set of each pair of classes

        G_, C_, b_: variables of different kind for definining the optimization problem
            These variables are precomputed in the `fit` method and are used for solving the optimization problem
            using `quadprog.solve_qp`. See `compute_ed_param_train` function

        a_: another variable of the optimization problem
            This one is computed in the `predict` method, just before solving the optimization problem

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
        Juan José del Coz: An analysis of quantification methods based on matching distributions
    """

    def __init__(self, estimator_train=None, estimator_test=None, distance=manhattan_distances, verbose=0):
        super(EDy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                  needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # variables to represent the distributions using the main idea of ED-based algorithms
        self.distance = distance
        self.train_n_cls_i_ = None
        self.train_distrib_ = None
        self.K_ = None
        #  variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.a_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit` method of its superclass.
            After that, the method computes all the elements of the optimization problem that involve just the
            training data:
            K_, G_, C_ and b_.

            Parameters
            ----------
            X: array-like, shape (n_examples, n_features)
                Data

            y: array-like, shape (n_examples, )
                True classes

            predictions_train: ndarray, shape (n_examples, n_classes) (probabilities)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Computing average distances for training distribution...' % self.__class__.__name__,
                  end='')

        n_classes = len(self.classes_)

        self.train_distrib_ = dict.fromkeys(self.classes_)
        self.train_n_cls_i_ = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            self.train_distrib_[cls] = self.predictions_train_[self.y_ext_ == cls]
            self.train_n_cls_i_[n_cls, 0] = len(self.train_distrib_[cls])

        self.K_, self.G_, self.C_, self.b_ = compute_ed_param_train(self.distance, self.train_distrib_,
                                                                    self.classes_, self.train_n_cls_i_)
        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the method computes a, the only element of the optimization problem that needs the testing
            data

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba` method)

                If predictions_test is not None they are copied on predictions_test_ and used.
                If predictions_test is None, predictions for the testing examples are computed using the `predict`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are at the same time None or not None

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        self.a_ = compute_ed_param_test(self.distance, self.train_distrib_, self.predictions_test_, self.K_,
                                        self.classes_, self.train_n_cls_i_)

        prevalences = solve_ed(G=self.G_, a=self.a_, C=self.C_, b=self.b_)

        if self.verbose > 0:
            print('done')

        return prevalences


class CvMy(UsingClassifiers):
    """ Multiclass CvMy method

        As described in (Castaño et al 2019), the predicted prevalences can be analytically calculated solving an
        optimization problem (with quadprog.solve_qp in this library). All ED-based methods share several functions
        in distribution_matching.utils. These functions are used to compute the elements of the optimization
        problem (`compute_ed_param_train`, `compute_ed_param_test`) and to solve the optimization problem (`solve_ed`)

        This class (as every other class based on distribution matching using classifiers) works in two different ways:

        1) Two estimators are used to classify training examples and testing examples in order to
           compute the distribution of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the fit/predict methods. This is useful
           for synthetic/artificial experiments

        The idea in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC and AC/PAC). In the first case, estimators are only trained once and can
        be shared for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the distribution of each class individually

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

        distance : distance function (default=manhattan_distances)
            It is the function used to compute the distance between every pair of examples

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimator_train and estimator_test could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because CvMy quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
             This means that predictions_train_/predictions_test_ contain probabilistic predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `predict` method whenever the true labels of the training set are needed, instead of y

        distance : distance function
            Function used to compute the distance between every pair of examples

        train_n_cls_i_ : ndarray, shape(n_classes, 1)
            Number of the examples of each class in the training set. Used to compute average distances

        train_distrib_ : Dict, the keys are the labels of the classes (classes_)
            Each key has associated a ndarray with the predictions, shape (train_n_cls_i_[i], 1) (binary quantification
            problems) or (train_n_cls_i_[i], n_classes) (multiclass quantification problems)

        test_distrib_ : ndarray, shape(n_examples, )
            The distribution of the test distribution

        K_ : ndarray, shape (n_classes, n_classes)
            Average distance between the examples in the training set of each pair of classes

        G_, C_, b_: variables of different kind for definining the optimization problem
            These variables are precomputed in the `fit` method and are used for solving the optimization problem
            using `quadprog.solve_qp`. See `compute_ed_param_train` function

        a_: another variable of the optimization problem
            This one is computed in the `predict` method, just before solving the optimization problem

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
        Juan José del Coz: An analysis of quantification methods based on matching distributions
    """

    def __init__(self, estimator_train=None, estimator_test=None, distance=manhattan_distances, verbose=0):
        super(CvMy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                   needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # variables to represent the distributions using the main idea of ED-based algorithms
        self.distance = distance
        self.train_n_cls_i_ = None
        self.train_distrib_ = None
        self.test_distrib_ = None
        self.K_ = None
        #  variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.a_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit` method of its superclass.
            After that, the method stores the true classes in y_train_ attribute.

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes) (probabilities)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            Then, the method computes all the elements of the optimization problem after computing the combined
            ranking of the predictions for the training examples and the testing examples using `rankdata` function

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba` method)

                If predictions_test is not None they are copied on predictions_test_ and used.
                If predictions_test is None, predictions for the testing examples are computed using the `predict`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are at the same time None or not None

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing average rankings for training and testing distribution...'
                  % self.__class__.__name__, end='')

        all_preds = np.concatenate([self.predictions_train_, self.predictions_test_])
        Hn = np.zeros(all_preds.shape)
        for i in range(Hn.shape[1]):
            Hn[:, i] = rankdata(all_preds[:, i])

        n_classes = len(self.classes_)

        Htr = Hn[:len(self.predictions_train_)]
        Htst = Hn[len(self.predictions_train_):]

        self.train_distrib_ = dict.fromkeys(self.classes_)
        self.train_n_cls_i_ = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            self.train_distrib_[cls] = Htr[self.y_ext_ == cls]
            self.train_n_cls_i_[n_cls, 0] = len(self.train_distrib_[cls])

        self.K_, self.G_, self.C_, self.b_ = compute_ed_param_train(self.distance, self.train_distrib_,
                                                                    self.classes_, self.train_n_cls_i_)

        if self.verbose > 0:
            print('done')
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        self.test_distrib_ = Htst

        self.a_ = compute_ed_param_test(self.distance, self.train_distrib_, self.test_distrib_, self.K_,
                                        self.classes_, self.train_n_cls_i_)

        prevalences = solve_ed(G=self.G_, a=self.a_, C=self.C_, b=self.b_)

        if self.verbose > 0:
            print('done')

        return prevalences


class EDX(WithoutClassifiers):
    """ Multiclass EDX method

        Parameters
        ----------
        distance: distance function (default=euclidean_distances)
            It is the function used to compute the distance between every pair of examples

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        distance_ : distance function
            The distance fuction used for computing the distance between every pair of examples

        classes_ : ndarray, shape (n_classes, )
            Class labels

        train_n_cls_i_ : ndarray, shape(n_classes, 1)
            Number of the examples of each class in the training set. Used to compute average distances

        train_distrib_ : Dict, the keys are the labels of the classes (classes_)
            Each key has associated a ndarray with the predictions, shape (train_n_cls_i_[i], 1) (binary quantification
            problems) or (train_n_cls_i_[i], n_classes) (multiclass quantification problems)

        K_ : ndarray, shape (n_classes, n_classes)
            Average distance between the examples in the training set of each pair of classes

        G_, C_, b_ : variables of different kind for definining the optimization problem
            These variables are precomputed in the `fit` method and are used for solving the optimization problem
            using `quadprog.solve_qp`. See `compute_ed_param_train` function

        verbose : int
            The verbosity level

        References
        ----------
        Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
        class-prior estimation under class balance change using energy distance. Transactions on Information
        and Systems 99, 1 (2016), 176–186.
    """
    def __init__(self, distance=euclidean_distances, verbose=0):
        super(EDX, self).__init__(verbose=verbose)
        # variables to represent the distributions using the main idea of ED-based algorithms
        self.distance_ = distance
        self.train_n_cls_i_ = None
        self.train_distrib_ = None
        self.K_ = None
        #  variables for solving the optimization problems
        self.G_ = None
        self.C_ = None
        self.b_ = None

    def fit(self, X, y):
        """ This method computes all the elements of the optimization that involve just the training data:
            K_, G_, C_ and b_.

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes
        """
        super().fit(X, y)

        if self.verbose > 0:
            print('Class %s: Computing average distances for training distribution...' % self.__class__.__name__,
                  end='')

        n_classes = len(self.classes_)

        self.train_distrib_ = dict.fromkeys(self.classes_)
        self.train_n_cls_i_ = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            self.train_distrib_[cls] = X[y == cls]
            self.train_n_cls_i_[n_cls, 0] = len(self.train_distrib_[cls])

        self.K_, self.G_, self.C_, self.b_ = compute_ed_param_train(self.distance_, self.train_distrib_,
                                                                    self.classes_, self.train_n_cls_i_)
        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X):
        """ Predict the class distribution of a testing bag

            This method computes a, the only element of the optimization problem that needs the testing
            data. Then, it solves the optimization problem using `quadprog.solve_qp` in `solve_ed` function

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        check_array(X, accept_sparse=True)

        a = compute_ed_param_test(self.distance_, self.train_distrib_, X, self.K_, self.classes_,
                                  self.train_n_cls_i_)

        prevalences = solve_ed(G=self.G_, a=a, C=self.C_, b=self.b_)

        if self.verbose > 0:
            print('done')

        return prevalences
