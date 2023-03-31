"""
Estimator object based on Cross Validation
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.exceptions import NotFittedError
from sklearn.model_selection._split import check_cv


class CV_estimator(BaseEstimator, ClassifierMixin):
    """ Cross Validation Estimator

        The idea is to have an estimator in which the model is formed by the models of a CV. This object is needed
        to estimate the distribution of the training set and testing sets. It has a `fit` method, that trains
        the models of the CV, and the typical methods `predict` y `predict_proba` to compute the predictions using
        such models. This implies that this object can be used by any distribution matching method that requires an
        estimator to represent the distributions

        Parameters
        ----------
        Mainly the same that `cross_validate` method in sklearn:

        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into train/test set.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if the estimator is a classifier and ``y`` is either binary or multiclass,
            :class:`StratifiedKFold` is used. In all other cases, :class:`KFold` is used.

        n_jobs : int or None, optional (default=None)
            The number of CPUs to use to do the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.

        fit_params : dict, optional
            Parameters to pass to the fit method of the estimator.

        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be
            useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
            This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

        averaged_predictions : bool, optional (default=True)
            If True, `predict`and `predict_proba` methods average the predictions given by estimators_ for
            each example

        voting : str, {'hard', 'soft'} (default='hard')
            Only used when averaged_predictions is True.
            If 'hard', `predict`and `predict_proba` methods apply majority rule voting.
            If 'soft', predict the class label based on the argmax of the sums of the predicted probabilities,
            which is recommended for an ensemble of well-calibrated classifiers.

        verbose : integer, optional (default=0)
            The verbosity level.

        Attributes
        ----------
        In addition to the parameters needed to call `cross_validate`` the class has this important attributes:

        estimator : An estimator object
            The estimator to fit each model of the CV

        estimators_ : list of trained estimators
            The list of estimators trained by `fit`method.
            The number of estimators is equal to the number of folds times number of repetitions

        averaged_predictions : bool
            Determines whether the predictions for each example given by estimators_ are averaged or not

        voting : str, {'hard', 'soft'} (default='hard')
            How predictions are aggregated:
                - 'hard', applying majority rule voting
                - 'soft', based on the argmax of the sums of the predicted probabilities

        le_ : a LabelEncoder fitted object
            Used to compute the class labels

        classes_ : ndarray, shape (n_classes, )
            Class labels

        X_train_ : array-like, shape (n_examples, n_features)
            Data. It is needed to obtain the predictions over the own training set

        y_train_ : array-like, shape (n_examples, )
            True classes. It is needed to obtain the predictions over the own training set

        verbose : integer
            The verbosity level.
    """

    def __init__(self, estimator, groups=None, cv='warn', n_jobs=None, fit_params=None, pre_dispatch='2*n_jobs',
                 averaged_predictions=True, voting='hard', verbose=0):
        self.estimator = estimator
        self.groups = groups
        self.cv = cv
        self.n_jobs = n_jobs
        self.fit_params = fit_params
        self.pre_dispatch = pre_dispatch
        self.averaged_predictions = averaged_predictions
        self.voting = voting
        self.verbose = verbose
        self.estimators_ = ()
        self.le_ = None
        self.classes_ = None
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        """ Fit the models
            It calls `cross_validate` to fit the models and save them in estimators_ attribute.
            It also stores some attributes needed by `predict` and `predict_proba`, namely, le_, classes_ X_train
            and y_train_

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes
        """
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.le_ = LabelEncoder().fit(y)
        # check cv
        self.cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        # train CV and save the estimators for each fold
        cvresults = cross_validate(self.estimator, X, y,
                                   groups=self.groups, cv=self.cv, n_jobs=self.n_jobs,
                                   verbose=self.verbose, fit_params=self.fit_params, pre_dispatch=self.pre_dispatch,
                                   return_estimator=True)
        self.estimators_ = cvresults['estimator']

        return self

    def predict(self, X):
        """ Returns the crisp predictions given by a CV estimator

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Test ata

            Returns
            -------
            preds : array-like, shape depends on two factors: the type of the examples (training or testing) and
                    the value of averaged_predictions attribute

                 Training set:
                     - averaged_predictions == True,  shape(n_examples, )
                     - averaged_predictions == False, shape(n_examples * n_reps, )
                 Testing set:
                     - averaged_predictions == True,  shape(n_examples, )
                     - averaged_predictions == False, shape(n_examples * n_reps * n_folds, )

                 Crisp predictions for the examples in X
        """
        if len(self.estimators_) == 0:
            raise NotFittedError('CV_estimator not fitted')

        preds = self._predict_proba(X)

        if self.averaged_predictions:
            if self.voting == 'soft':
                preds = np.mean(preds, axis=0)
                preds = np.argmax(preds, axis=1)
                preds = self.le_.inverse_transform(preds)
            else:
                # hard
                #  for each example (axis=2), compute the class with the largest probability
                aux = np.apply_along_axis(np.argmax, axis=2, arr=preds)
                # compute the number of votes for each class
                aux2 = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(self.classes_)), axis=0, arr=aux)
                # compute the class with more votes
                aux3 = np.argmax(aux2, axis=0)
                # transforming label position to class label
                preds = self.le_.inverse_transform(aux3)
        else:
            # not averaging, so each pred is treated independently
            preds = preds.reshape(-1, len(self.classes_))
            # computing the class with largest probability
            preds = np.argmax(preds, axis=1)
            # transforming label position to class label
            preds = self.le_.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        """ Returns probabilistic predictions given by a CV estimator

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Test ata

            Returns
            -------
            preds : array-like, shape depends on two factors: the type of the examples (training or testing) and
                    the value of averaged_predictions attribute

                Probabilistic predictions for the examples in X.
                Shape:
                    Training set:
                        - averaged_predictions == True,  shape(n_examples, n_classes)
                        - averaged_predictions == False, shape(n_examples * n_reps, n_classes)
                    Testing set:
                        - averaged_predictions == True,  shape(n_examples, n_classes)
                        - averaged_predictions == False, shape(n_examples * n_reps * n_folds, n_classes)
        """
        if len(self.estimators_) == 0:
            raise NotFittedError('CV_estimator not fitted')

        preds = self._predict_proba(X)

        if self.averaged_predictions:
            preds = np.mean(preds, axis=0)
        else:
            preds = preds.reshape(-1, len(self.classes_))
        return preds

    def _predict_proba(self, X):
        """ Returns all probabilistic predictions given by a CV estimator

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Test ata

            Returns
            -------
            preds : array-like, shape (nreps, n_examples, n_classes) for the training set or
                    (n_reps*n_folds, n_examples, n_classes) for a testing bag

                Raw probabilistic predictions for all the example of X. Each example may have more than one prediction

            When X is equal to the training set (X_train) each example is just predicted by those estimators not
            trained with that example.
            When X is a test bag, each examples is predicted by all estimators
        """
        n_examples = len(X)
        if np.array(X == self.X_train_).all():
            # predicting over training examples, same partitions
            #  computing number of repetitions
            n_preds = 0
            for (train_index, test_index) in self.cv.split(self.X_train_, self.y_train_):
                n_preds = n_preds + len(test_index)
            n_repeats = n_preds // n_examples
            #  storing predictions
            preds = np.zeros((n_repeats, n_examples, len(self.classes_)), dtype=float)
            n_rep = 0
            n_preds = 0
            for nfold, (train_index, test_index) in enumerate(self.cv.split(self.X_train_, self.y_train_)):
                X_test = X[test_index]
                preds[n_rep, test_index, :] = self.estimators_[nfold].predict_proba(X_test)
                n_preds = n_preds + len(test_index)
                if n_preds == n_examples:
                    n_rep += 1
                    n_preds = 0
        else:
            #   it is a test sample, predicting with each estimator
            preds = np.zeros((len(self.estimators_), n_examples, len(self.classes_)), dtype=float)
            for n_est, est in enumerate(self.estimators_):
                preds[n_est, :, :] = est.predict_proba(X)

        return preds
