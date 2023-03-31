"""
Quantifier based on Mixture Estimation proposed by Friedman
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from quantificationlib.base import UsingClassifiers
from quantificationlib.search import global_search, mixture_of_pdfs
from quantificationlib.optimization import solve_hd, compute_l2_param_train, solve_l2, solve_l1


class FriedmanME(UsingClassifiers):
    """ Multiclass Mixture Estimation method proposed by Friedman

        This class works in two different ways:

        1) Two estimators are used to classify the examples of the training set and the testing set in order to
           compute the (probabilistic) confusion matrix of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the `fit`/`predict methods. This is useful
           for synthetic/artificial experiments

        The idea in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC). In the first case, estimators are only trained once and can be shared
        for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the confusion matrix

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to obtain the confusion matrix of the testing set

        distance : str, representing the distance function (default='L2')
            It is the name of the distance used to compute the difference between the mixture of the training
            distribution and the testing distribution

        tol : float, (default=1e-05)
            The precision of the solution when search is used to compute the prevalence

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimators could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because FriedmanME quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
             This means that predictions_train_/predictions_test_ contain probabilistic predictions

        distance : A distance function (default=l2)
            The name of the distance function used

        tol : float
            The precision of the solution when search is used to compute the solution

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit` method whenever the true labels of the training set are needed, instead of y

        train_prevs_ : ndarray, shape (n_classes, )
            Prevalence of each class in the training set

        train_distrib_ : ndarray, shape (n_classes, n_classes)
            Each column is the representation of the training examples of such class. The column contains the
            percentage of examples of each class whose probability to belong to the row class is >= than
            the prevalence of the row class in the training set

        test_distrib_ : ndarray, shape (n_classes_, 1)
            Percentage of examples in the testing bag whose probability to belong each class is >= than
            the prevalence of that class in the training set

        G_, C_, b_: variables of different kind for definining the optimization problem
            These variables are precomputed in the `fit` method and are used for solving the optimization problem
            using `quadprog.solve_qp`. See `compute_l2_param_train` function

        problem_ : a cvxpy Problem object
            This attribute is set to None in the fit() method. With such model, the first time a testing bag is
            predicted this attribute will contain the corresponding cvxpy Object (if such library is used, i.e in the
            case of 'L1' and 'HD'). For the rest testing bags, this object is passed to allow a warm start. The
            solving process is faster.

        mixtures_ : ndarray, shape (101, n_quantiles)
            Contains the mixtures for all the prevalences in the range [0, 1] step=0.01. This speeds up the prediction
            for a collection of testing bags

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Jerome H. Friedman. Class counts in future unlabeled samples. Presentation at MIT CSAIL Big Data Event, 2014.
    """

    def __init__(self, estimator_test=None, estimator_train=None, distance='L2', tol=1e-05, verbose=0):
        super(FriedmanME, self).__init__(estimator_test=estimator_test, estimator_train=estimator_train,
                                         needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # parameters
        self.distance = distance
        self.tol = tol
        # variables to represent the distributions
        self.train_distrib_ = None
        self.test_distrib_ = None
        self.train_prevs_ = None
        # variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.problem_ = None
        self.mixtures_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit method of its superclass.
            Then, the method computes the training distribution of the method ME suggested by Friedman. The distribution
            of a class contains the percentage of the training examples of that class whose probability to belong
            to each class is >= than the prevalence of such class in the training set
            Finally, the method computes all the parameters for solving the optimization problem needed by quadprog
            that do not need the testing distribution

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating training distribution...' % self.__class__.__name__,
                  end='')

        n_classes = len(self.classes_)

        self.train_prevs_ = np.unique(y, return_counts=True)[1] / len(y)

        #   for each pair (example, class), this matrix has a 1 if the predicted probability that the example belongs
        #   to that class is >= than the prevalence of that class in the training set
        Vp = np.zeros((len(self.predictions_train_), n_classes))
        for n_cls in range(n_classes):
            Vp[:, n_cls] = np.array(self.predictions_train_[:, n_cls] >= self.train_prevs_[n_cls]).astype(int)

        self.train_distrib_ = np.zeros((n_classes, n_classes))
        for n_cls, cls in enumerate(self.classes_):
            self.train_distrib_[:, n_cls] = Vp[self.y_ext_ == cls].mean(axis=0)

        if self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.train_distrib_, self.classes_)

        if self.verbose > 0:
            print('done')

        self.problem_ = None
        self.mixtures_ = None

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict() method.

            After that, the method computes the distribution of the FriedmanME method for the testing bag. That is,
            the percentage of examples in the testing bag whose probability to belong each class is >= than
            the prevalence of that class in the training set

            Finally, the prevalences are computed solving the resulting optimization problem

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

        Up = np.zeros((len(self.predictions_test_), n_classes))
        for n_cls in range(n_classes):
            Up[:, n_cls] = np.array(self.predictions_test_[:, n_cls] >= self.train_prevs_[n_cls]).astype(int)

        self.test_distrib_ = Up.mean(axis=0).reshape(-1, 1)

        if self.distance == 'HD':
            self.problem_, prevalences = solve_hd(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                                  n_classes=n_classes, problem=self.problem_)
        elif self.distance == 'L2':
            prevalences = solve_l2(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   G=self.G_, C=self.C_, b=self.b_)
        elif self.distance == 'L1':
            self.problem_, prevalences = solve_l1(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                                  n_classes=n_classes, problem=self.problem_)
        else:
            self.mixtures_, prevalences = global_search(distance_func=self.distance, mixture_func=mixture_of_pdfs,
                                                        test_distrib=self.test_distrib_, tol=self.tol,
                                                        mixtures=self.mixtures_, return_mixtures=True,
                                                        pos_distrib=self.train_distrib_[:, 1].reshape(-1, 1),
                                                        neg_distrib=self.train_distrib_[:, 0].reshape(-1, 1))
        if self.verbose > 0:
            print('done')

        return prevalences
