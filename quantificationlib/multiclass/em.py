import numpy as np

from quantificationlib.base import UsingClassifiers
from quantificationlib.metrics.multiclass import mean_absolute_error


class EM(UsingClassifiers):
    """ EM method

        The main idea of this method is to estimate the prevalences applying EM algorithm.

        This class (as every other class based on using a classifier) works in two different ways:

        1) Two estimators are used to classify training examples and testing examples in order to
           compute the distribution of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the fit/predict methods. This is useful
           for synthetic/artificial experiments

        The goal in both cases is to guarantee that all methods based on classifier are using **exactly**
        the same predictions when you compare this kind of quantifiers. In the first case, estimators are only trained
        once and can be shared for several quantifiers of this kind

        Theoretically, it also works for multiclass quantification.

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the distribution of each class individually

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        epsilon: float, (default=1e-04)
            EM algorithm (its loop) stops when the difference between the prevalences of two
            consecutive iterations is lower than epsilon

        max_iter: int (default=1000)
            The maximum number of iterations for the loop of the EM algorithm

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

        verbose : int
            The verbosity level

        epsilon_: float
            EM algorithm (its loop) stops when the mean absolute error between the prevalences of two
            consecutive iterations is lower than epsilon

        max_iter_: int
            The maximum number of iterations for the loop of the EM algorithm

        prevalences_train_ : ndarray, shape (n_classes, )
            Prevalence of each class in the training dataset

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Marco Saerens, Patrice Latinne, Christine Decaestecker: Adjusting the outputs of a classifier to new a priori
        probabilities: a simple procedure. Neural computation, 14(1) (2002), 21-41.
    """

    def __init__(self, estimator_train=None, estimator_test=None, verbose=0, epsilon=1e-4, max_iter=1000):
        super(EM, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                 needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        self.epsilon_ = epsilon
        self.max_iter_ = max_iter
        self.prevalences_train_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit` method of its superclass.

            After that, the method just computes the prevalences in the training set

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

        n_classes = len(self.classes_)

        freq = np.zeros(n_classes)
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls] = np.equal(y, cls).sum()

        self.prevalences_train_ = freq / float(len(y))

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the method applies EM algorithm. In the E step the new posterior are computed based on the
            estimated prevalences

                    P_tst(y|x) = P_tst(y)/P_tr(y) P_tr(y|x)

            and are normalized to sum 1
            Then, in the M step new prevalences are computed as the mean of the posteriors obtained in the E step

            The loop stops when the max number of iterations is reached or when the difference between the
            prevalences of two consecutive iterations is lower than epsilon

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
            print('Class %s: Estimating prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        iterations = 0
        prevalences = np.copy(self.prevalences_train_)
        prevalences_prev = np.ones(n_classes)

        while iterations < self.max_iter_ and (mean_absolute_error(prevalences, prevalences_prev) > self.epsilon_
                                               or iterations < 10):

            nonorm_posteriors = np.multiply(self.predictions_test_, np.divide(prevalences, self.prevalences_train_))

            posteriors = np.divide(nonorm_posteriors, nonorm_posteriors.sum(axis=1, keepdims=True))

            prevalences_prev = prevalences
            prevalences = posteriors.mean(0)

            iterations = iterations + 1

        if self.verbose > 0:
            if iterations < self.max_iter_:
                print('done')
            else:
                print('done but it might have not converged, max_iter reached')

        return prevalences
