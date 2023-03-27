import numpy as np

from quantificationlib.base import UsingClassifiers
from quantificationlib.search import golden_section_search, compute_sord_weights, sord


class SORDy(UsingClassifiers):
    """ SORDy method

        The idea is to represent the mixture of the training distribution and the testing distribution using
        the whole set of predictions given by a classifier (y).

        This class (as every other class based on distribution matching using classifiers) works in two different ways:

        1) Two estimators are used to classify training examples and testing examples in order to
           compute the distribution of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the fit/predict methods. This is useful
           for synthetic/artificial experiments

        The goal in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC and AC/PAC). In the first case, estimators are only trained once and can
        be shared for several quantifiers of this kind

        Multiclass quantification is not implemented yet for this object. It would need a more complex searching
        algorithm (instead golden_section_search)

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the distribution of each class individually

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

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
            It is True because SORD quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
            This means that predictions_train_/predictions_test_ contain probabilistic predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit`/`predict` method whenever the true labels of the training set are needed,
            instead of y

        tol : float
            The precision of the solution when search is used to compute the solution

        train_distrib_ : ndarray, shape (n_examples_train, 1) binary quantification
            Contains predictions_train_ in ascending order

        train_labels_ : ndarray, shape (n_examples_train, 1) binary quantification
            Contains the corresponding labels of the examples in train_distrib_ in the same order

        union_distrib_ : ndarray, shape (n_examples_train+n_examples_test, 1)
            Contains the union of predictions_train_ and predictions_test_ in ascending order

        union_labels  :  ndarray, shape (n_examples_train+n_examples_test, 1)
            Contains the set/or  the label of each prediction  in union_distrib_. If the prediction corresponds to
            a training example, the value is the true class of such example. If the example belongs to the testing
            distribution, the value is NaN

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        André Maletzke, Denis dos Reis, Everton Cherman, and Gustavo Batista: Dys: A framework for mixture models
        in quantification. In AAAI 2019, volume 33, pp. 4552–4560. 2019.
    """

    def __init__(self, estimator_train=None, estimator_test=None, tol=1e-05, verbose=0):
        super(SORDy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                    needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # attributes
        self.tol = tol
        # variables to represent the distributions
        self.train_distrib_ = None
        self.train_labels_ = None
        self.union_distrib_ = None
        self.union_labels_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit` method of its superclass.
            After that, the method sorts the predictions for the train set into train_distrib_. The actual
            labels are put in the same order in train_labels_

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
                When estimator_train and predictions_train are at the same time None or not None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if len(self.classes_) > 2:
            raise TypeError("SORDy is a binary method, multiclass quantification is not supported")

        if self.verbose > 0:
            print('Class %s: Collecting data from training distribution...' % self.__class__.__name__, end='')

        #   sorting the probabilities of belonging to the positive class, P(y=+1 | x)
        indx = np.argsort(self.predictions_train_[:, 1])
        self.train_distrib_ = self.predictions_train_[indx, 1]
        self.train_labels_ = self.y_ext_[indx]

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the method computes the union distribution, merging training and testing distributions. This
            union is sorted based on the posterior probability of each example.

            Finally, the prevalences are computed using golden section search using two functions:
                -compute_sord_weights: to obtain the mixture depending on the estimated prevalence
                -sord: distance funtion to measure how similar the mixture and the testing distributions are

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba` method)

                If estimator_test is None then predictions_test can not be None.
                If predictions_test is None, predictions for the testing examples are computed using the `predict_proba`
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
            print('Class %s: Computing the testing distribution...' % self.__class__.__name__, end='')

        #  sorting the probabilities of belonging to the positive class, P(y=+1 | x)
        self.union_distrib_ = np.hstack((self.train_distrib_, self.predictions_test_[:, 1]))
        indx = np.argsort(self.union_distrib_)
        self.union_distrib_ = self.union_distrib_[indx]
        test_label = np.max(self.classes_)+1
        self.union_labels_ = np.hstack((self.train_labels_,
                                        np.full(len(self.predictions_test_[:, 1]), test_label)))
        self.union_labels_ = self.union_labels_[indx]

        if self.verbose > 0:
            print('done')
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        _, prevalences = golden_section_search(distance_func=sord, mixture_func=compute_sord_weights,
                                               test_distrib=self.union_distrib_, tol=self.tol, a=0.0, b=1.0,
                                               union_labels=self.union_labels_,
                                               classes=self.classes_)
        if self.verbose > 0:
            print('done')

        return prevalences
