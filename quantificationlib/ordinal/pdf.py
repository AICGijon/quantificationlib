"""
Ordinal versions for quantifiers based on representing the distributions using PDFs
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from quantificationlib.base import UsingClassifiers
from quantificationlib.optimization import solve_hd, compute_l2_param_train, solve_l2, solve_l1


class PDFOrdinaly(UsingClassifiers):
    """ Generic Ordinal version of PDFy method

        The idea is to represent the mixture of the training distribution and the testing distribution using PDFs of
        the predictions given by a classifier (y). The difference between both is minimized using a distance/loss
        function. Originally, (González et al. 2013) propose the Hellinger Distance, but any other distance/loss
        function could be used, like L1 or L2. The class has a parameter to select the distance used.

        This class (as all classes based on distribution matching using classifiers) works in two different ways:

        1) Two estimators are used to classify training examples and testing examples in order to
           compute the distribution of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the fit/predict methods. This is useful
           for synthetic/artificial experiments

        The goal in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers. In the first case, estimators are only
        trained once and can be shared for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the distribution of each class individually

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

        n_bins : int
            Number of bins to compute the PDFs

        distance : str, representing the distance function (default='HD')
            It is the name of the distance used to compute the difference between the mixture of the training
            distribution and the testing distribution

        method : str
            'full_probabilistic' predictions for training and testing set contain a probability for each class
            'winner_node' this method is only applicable for DDAGClassifier, and predictions contain just the
                          probabilities for the two consecutive classes of the winner node previous to the leaves

        tol : float, (default=1e-05)
            The precision of the solution when search is used to compute the prevalence

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
            It is True because PDFy quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
             This means that predictions_train_/predictions_test_ contain probabilistic predictions

        distance : str or a distance function
            A string with the name of the distance function ('HD'/'L1'/'L2') or a distance function

        tol : float
            The precision of the solution when search is used to compute the solution

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit` method whenever the true labels of the training set are needed, instead of y

        n_bins : int (default=8)
            The number of bins to compute the PDFs

        train_distrib_ : ndarray, shape (n_bins * 1, n_classes) binary or (n_bins * n_classes_, n_classes) multiclass
            The PDF for each class in the training set

        test_distrib_ : ndarray, shape (n_bins * 1, 1) binary quantification or (n_bins * n_classes_, 1) multiclass
            The PDF for the testing bag

        G_, C_, b_: variables of different kind for defining the optimization problem
            These variables are precomputed in the `fit` method and are used for solving the optimization problem
            using `quadprog.solve_qp`. See `compute_l2_param_train` function

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
    """

    def __init__(self, estimator_train=None, estimator_test=None, n_bins=8, distance='L2',
                 method='full_probabilistic', tol=1e-05, verbose=0):
        super(PDFOrdinaly, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                          needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # attributes
        self.n_bins = n_bins
        self.distance = distance
        self.method = method
        self.tol = tol
        # variables to represent the distributions
        self.train_distrib_ = None
        self.test_distrib_ = None
        # variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.problem_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit` method of its superclass.
            After that, the method computes the pdfs for all the classes in the training set

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating PDFs for training distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)

        if len(self.classes_) <= 2:
            raise TypeError("PDFOrdinaly is a ordinal method, the number of classes must be greater than 2")

        if self.method == 'full_probabilistic':
            scores = self.predictions_train_.dot(np.array(range(1, n_classes + 1)))
        elif self.method == 'winner_node':
            scores = self.__leafprobs2scores(self.predictions_train_, n_classes)
        else:
            raise ValueError('Method %s not implemented with objects of class %s', self.method, self.__class__.__name__)

        effective_n_bins = self.n_bins * (n_classes - 1)
        self.train_distrib_ = np.zeros((effective_n_bins, n_classes))
        for n_cls, cls in enumerate(self.classes_):
            self.train_distrib_[:, n_cls] = \
                   np.histogram(scores[self.y_ext_ == cls], bins=effective_n_bins, range=(1., float(n_classes)))[0]
            # compute pdf
            self.train_distrib_[:, n_cls] = self.train_distrib_[:, n_cls] / (np.sum(self.y_ext_ == cls))

        if self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.train_distrib_, self.classes_)
        elif self.distance == 'EMD':
            for i in range(1, effective_n_bins):
                self.train_distrib_[i, :] = np.sum(self.train_distrib_[i-1:i+1, :], axis=0)

        if self.verbose > 0:
            print('done')

        self.problem_ = None

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the method computes the PDF for the testing bag.

            Finally, the prevalences are computed using the corresponding function according to the value of
            distance attribute

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

        if self.method == 'full_probabilistic':
            scores = self.predictions_test_.dot(np.array(range(1, n_classes + 1)))
        elif self.method == 'winner_node':
            scores = self.__leafprobs2scores(self.predictions_test_, n_classes)
        else:
            raise ValueError('Method %s not implemented with objects of class %s', self.method, self.__class__.__name__)

        effective_n_bins = self.n_bins * (n_classes - 1)
        self.test_distrib_ = np.histogram(scores, bins=effective_n_bins, range=(1., float(n_classes)))[0]
        self.test_distrib_ = self.test_distrib_ / len(self.predictions_test_)

        if self.distance == 'HD':
            self.problem_, prevalences = solve_hd(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                                  n_classes=n_classes, problem=self.problem_)
        elif self.distance == 'L2':
            prevalences = solve_l2(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   G=self.G_, C=self.C_, b=self.b_)
        elif self.distance == 'EMD':
            for i in range(1, effective_n_bins):
                self.test_distrib_[i] = self.test_distrib_[i] + self.test_distrib_[i-1]
            self.problem_, prevalences = solve_l1(train_distrib=self.train_distrib_[0:effective_n_bins-1, :],
                                                  test_distrib=self.test_distrib_[0:effective_n_bins-1],
                                                  n_classes=n_classes, problem=self.problem_)
            # clipping the prevalences according to (Forman 2008)
            prevalences = np.clip(prevalences, 0, 1)

            if np.sum(prevalences) > 0:
                prevalences = prevalences / float(np.sum(prevalences))
        else:
            prevalences = None

        if self.verbose > 0:
            print('done')

        return prevalences

    @staticmethod
    def __leafprobs2scores(probs, n_classes):
        indexes = np.argmax(probs > 0, axis=1)
        indexes[indexes == n_classes - 1] = n_classes - 2
        indexes = indexes + 1
        scores = np.zeros(len(probs))
        for n_row, n_col in enumerate(indexes):
            scores[n_row] = n_col + probs[n_row, n_col]
        return scores
