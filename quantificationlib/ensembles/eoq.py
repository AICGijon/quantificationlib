import numpy as np

from copy import deepcopy

from sklearn.utils import check_X_y
from sklearn.exceptions import NotFittedError

from scipy.spatial.distance import cdist

from quantificationlib.base import BaseQuantifier, UsingClassifiers, WithoutClassifiers
from quantificationlib.bag_generator import BagGenerator, PriorShift_BagGenerator
from quantificationlib.estimators.ensembles import EnsembleOfClassifiers

from quantificationlib.multiclass.df import compute_bincuts

from quantificationlib.metrics.multiclass import mean_absolute_error


class EoQ(WithoutClassifiers):
    """ This class implements Ensembles of Quantifiers for all kind of quantifiers. All the quantifiers of the
        ensemble are of the same class and using the same parameters.

        Parameters
        ----------
        base_quantifier : quantifier object, optional, (default=None)
            The quantifier used for each model of the ensemble

        n_quantifiers: int, (default=100)
            Number of quantifiers in the ensemble

        bag_generator : BagGenerator object (default=PriorShift_BagGenerator())
            Object to generate the bags (with a selected shift) for training each quantifier

        combination_strategy: str, (default='mean')
            Strategy used to combine the predictions of the quantifiers

        ensemble_estimator_train : estimator object, optional, (default=None)
            Estimator used to classify the examples of the training bags when a base_quantifier of
            class UsingClassifiers is used. A regular estimator can be used, this implies that a unique classifier is
            share for all the quantifiers in the ensemble. If the users prefers that each quantifier uses an
            individual classifier the an estimator of the class EnsembleOfClassifiers must be passed here

        ensemble_estimator_test : estimator object, optional, (default=None)
            Estimator used to classify the examples of the testing bags. A regular estimator can be used, this implies
            that a unique classifier is share for all the quantifiers in the ensemble. If the users prefers that each
            quantifier uses an individual classifier the an estimator of the class EnsembleOfClassifiers must be passed
            here

        distribution_function : str, (default='PDF')
            Method to estimate the distributions of training and testing bags. Possible values 'PDF' or 'CDF'.
            This is used just for distribution_similarity combination strategy. This strategy is based on comparing
            the PDFs or CDFs of the training bags and the PDF/CDF of the testing bag, selecting those quantifiers
            training over the most similar distributions. To compute the distribution, EoQ employs the input
            features (Xs) for quantifiers derived from WithoutClassifiers class and the predictions (Ys) for
            quantifiers derived from UsingClassifiers

        n_bins : int, (default=100)
            Numbers of bins to estimate the distributions of training and testing bags. This is needed for
            distribution_similarity combination strategy.

        bin_strategy = str, (default='equal_width')
            Method to compute the boundaries of the bins for to estimate the distributions of training and testing
            bags when the distribution_similarity combination strategy is used. Possible values:
                'equal_width': bins of equal length (it could be affected by outliers)
                'equal_count': bins of equal counts (considering the examples of all classes)
                'binormal': (Only for binary quantification) It is inspired on the method devised by
                          (Tasche, 2019, Eq (A16b)). the cut points, $-\infty < c_1 < \ldots < c_{b-1} < \infty$,
                          are computed as follows based on the assumption that the features follow a normal distribution:

                          $c_i = \frac{\sigma^+ + \sigma^{-}}{2} \ \Phi^{-1}\bigg(\frac{i}{b}\bigg)  + \frac{\mu^+ + \mu^{-}}{2} ,  \quad i=1,\ldots,b-1$

                          where $\Phi^{-1}$ is the quantile function of the standard normal distribution, and $\mu$
                          and $\sigma$ of the normal distribution are estimated as the average of those values for
                          the training examples of each class.

                'normal':  The idea is that each feacture follows a normal distribution. $\mu$ and $\sigma$ are
                           estimated as the weighted mean and std from the training distribution. The cut points
                           $-\infty < c_1 < \ldots < c_{b-1} < \infty$ are computed as follows:

                           $c_i = \sigma^ \ \Phi^{-1}\bigg(\frac{i}{b}\bigg)  + \mu ,  \quad i=1,\ldots,b-1$

        distance_bags : str, (default='euclidean')
            Distance used to compute distribution similarity

        percentage_of_selected_models : float, value in [0, 1], (default=0.5)
            Percentage of selected models for distribution similarity and prevalence similarity strategies

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode


        Attributes
        ----------
        base_quantifier : quantifier object
            The quantifier used for each model of the ensemble

        n_quantifiers: int
            Number of quantifiers in the ensemble

        bag_generator : BagGenerator object
            Object to generate the bags for training each quantifier

        combination_strategy: str
            Strategy used to combine the predictions of the ensemble quantifiers

        ensemble_estimator_train : estimator object
            Estimator used to classify the examples of the training bags when a base_quantifier of
            class UsingClassifiers is used

        ensemble_estimator_test : estimator object
            Estimator used to classify the examples of the testing bags

        distribution_function : str
            Method to estimate the distributions of training and testing bags

        n_bins : int
            Numbers of bins to estimate the distributions of training and testing bags

        bin_strategy = str, (default='equal_width')
            Method to compute the boundaries of the bins for to estimate the distributions of training and testing
            bags

        distance_bags : str
            Distance used to compute distribution similarity

        percentage_of_selected_models : float
            Percentage of selected models for distribution similarity and prevalence similarity strategies

        quantifiers_ : ndarray, shape (n_quantifiers,)
            This vector stores the quantifiers of the ensemble

        prevalences_ : ndarray, shape (n_quantifiers,)
            It contains the prevalence of each training bag used to fit each quantifier of the ensemble

        indexes_ : ndarry, shape (n_examples_of_training_bags, n_quantifiers)
            The indexes of the training examples that compose each training bag. The number of training examples
            used in each bag is fixed true bag_generator parameter

        bincuts_ : ndarray, shape (n_features, n_bins + 1)
            Bin cuts for each feature used to  estimate the training/testing distributions for distribution
            similarity strategy. The total number of features depends on the kind of base_quantifier used and on
            the quantification problem. For quantifiers derived from WithoutClassifiers n_features is the dimension on
            the input space. For quantifiers derived from UsingClassifiers n_features is 1 for binary quantification
            tasks and is n_classes for multiclass/ordinal problems

        distributions_ : ndarray, shape (n_quantifiers, n_features * n_bins)
            It constains the estimated distribution for each quantifier

        classes_ : ndarray, shape (n_classes, )
            Class labels

        verbose : int
            The verbosity level
    """
    def __init__(self, base_quantifier=None, n_quantifiers=100, bag_generator=PriorShift_BagGenerator(),
                 combination_strategy='mean',
                 ensemble_estimator_train=None, ensemble_estimator_test=None,
                 distribution_function='PDF', n_bins=100, bin_strategy='equal_width', distance_bags='euclidean',
                 percentage_of_selected_models=0.5,
                 verbose=0):
        # attributes
        self.base_quantifier = base_quantifier
        self.n_quantifiers = n_quantifiers
        self.bag_generator = bag_generator
        self.combination_strategy = combination_strategy
        self.ensemble_estimator_train = ensemble_estimator_train
        self.ensemble_estimator_test = ensemble_estimator_test
        self.distribution_function = distribution_function
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.distance_bags = distance_bags
        self.percentage_of_selected_models = percentage_of_selected_models
        self.verbose = verbose
        #  variables to store the quantifiers
        self.quantifiers_ = None
        #  variables to represent the training bags
        self.prevalences_ = None
        self.indexes_ = None
        #  variables to represent information about the training bags
        self.bincuts_ = None
        self.distributions_ = None
        #  other variables
        self.classes_ = None

    def fit(self, X, y, predictions_train=None, prevalences=None, indexes=None):
        """ This method does the following tasks:

            1) It generates the training bags using a Bag_Generator object
            2) It fits the quantifiers of the ensemble.
            In the case of quantifiers derived from the class UsingClassifiers, there are 3 possible ways to do this:
            - train a classifier for each bag. To do this an object from the class EnsembleOfClassifiers must be
              passed on ensemble_estimator
            - train a classifier for the whole training set using an estimator from other class
            - uses the predictions_train given in the predictions_train parameter (these predictions usually are
              obtained applying an estimator over the whole training set like in the previous case)

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, optional, shape(n_examples, 1) crisp or shape (n_examples, n_classes) (probs
                                with a regular estimator) or shape(n_examples, n_estimators, n_classes) with an
                                instance of EnsembleOfClassifiers
                Predictions of the examples in the training set

            prevalences : array-like, shape (n_classes, n_bags)
                i-th row contains the true prevalences of each bag

            indexes : array-line, shape (bag_size, n_bags)
                i-th column contains the indexes of the examples for i-th bag
        """
        if not isinstance(self.base_quantifier, BaseQuantifier):
            raise ValueError("Class %s: quantifier must be an object of a class derived from BaseQuantifier"
                             % self.__class__.__name__)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if self.verbose > 0:
            print('Class %s: Creating training bags...' % self.__class__.__name__, end='')

        #  compute bags if needed
        if prevalences is None and indexes is None:
            if not isinstance(self.bag_generator, BagGenerator):
                raise ValueError("Invalid bag generaror object")

            self.bag_generator.n_bags = self.n_quantifiers
            self.prevalences_, self.indexes_ = self.bag_generator.generate_bags(X, y)
        else:
            self.prevalences_ = prevalences
            self.indexes_ = indexes

        #  fit quantifiers
        if isinstance(self.base_quantifier, WithoutClassifiers):

            if self.verbose > 0:
                print('Class %s: Training quantifiers...' % self.__class__.__name__, end='')

            self.bincuts_ = np.zeros((X.shape[1], 100 + 1))
            for att in range(X.shape[1]):
                self.bincuts_[att, :] = compute_bincuts(x=X[:, att], y=None, classes=self.classes_,
                                                        n_bins=self.n_bins, bin_strategy='equal_width')

            self.quantifiers_ = np.zeros(self.n_quantifiers, dtype=object)
            for i in range(self.n_quantifiers):
                self.quantifiers_[i] = deepcopy(self.base_quantifier)
                self.quantifiers_[i].fit(X[self.indexes_[:, i], :], y[self.indexes_[:, i]])

            #  compute distributions for distribution_similarity combination strategy
            self.distributions_ = np.zeros((self.n_quantifiers, self.n_bins * X.shape[1]))
            for i in range(self.n_quantifiers):
                for att in range(X.shape[1]):
                    self.distributions_[i, att * self.n_bins:(att + 1) * self.n_bins] = \
                        np.histogram(X[:, att], bins=self.bincuts_[att, :])[0] / len(X)
                    if self.distribution_function == 'CDF':
                        self.distributions_[i, att * self.n_bins:(att + 1) * self.n_bins] = \
                            np.cumsum(self.distributions_[i, att * self.n_bins:(att + 1) * self.n_bins], axis=1)

        elif isinstance(self.base_quantifier, UsingClassifiers):

            if predictions_train is None:

                if self.ensemble_estimator_train is None:
                    raise ValueError("ensemble_estimator_train or predictions_train must be not None "
                                     "with objects of class %s", self.__class__.__name__)

                #  fit train estimator
                if self.verbose > 0:
                    print('Class %s: Fitting ensemble estimator for training distribution...'
                          % self.__class__.__name__, end='')

                try:
                    self.ensemble_estimator_train.predict(X[0:1, :].reshape(1, -1))
                    if self.verbose > 0:
                        print('it was already fitted')

                except NotFittedError:

                    X, y = check_X_y(X, y, accept_sparse=True)

                    #  if ensemble_estimator_train is an instance of EnsembleOfClassfier we need to pass the indexes
                    #  of the instances of each bag
                    if isinstance(self.ensemble_estimator_train, EnsembleOfClassifiers):
                        self.ensemble_estimator_train.fit(X, y, self.indexes_)
                    else:
                        self.ensemble_estimator_train.fit(X, y)

                    if self.verbose > 0:
                        print('fitted')

                #  compute predictions_train
                if self.base_quantifier.probabilistic_predictions:
                    predictions_train = self.ensemble_estimator_train.predict_proba(X)
                else:
                    predictions_train = self.ensemble_estimator_train.predict(X)

            if self.ensemble_estimator_test is not None:
                if self.verbose > 0:
                    print('Class %s: Fitting ensemble estimator for testing distribution...'
                          % self.__class__.__name__, end='')

                # we need to fit the estimator for the testing distribution
                # we check if the estimator is trained or not
                try:
                    self.ensemble_estimator_test.predict(X[0:1, :].reshape(1, -1))
                    if self.verbose > 0:
                        print('it was already fitted')

                except NotFittedError:

                    X, y = check_X_y(X, y, accept_sparse=True)

                    #  if ensemble_estimator_test is an instance of EnsembleOfClassfier we need to pass the indexes
                    #  of the instances of each bag
                    if isinstance(self.ensemble_estimator_test, EnsembleOfClassifiers):
                        self.ensemble_estimator_test.fit(X, y, self.indexes_)
                    else:
                        self.ensemble_estimator_test.fit(X, y)

                    if self.verbose > 0:
                        print('fitted')

            # Compute y_ext and indexes_ext (this may be needed when CV_estimator is used and we have several
            # predictions for each example
            if len(y) == len(predictions_train):
                y_ext = y
                indexes_ext = self.indexes_
            else:
                n_repetitions = len(predictions_train) // len(y)
                y_ext = np.tile(y, n_repetitions)
                adapt_indexs = np.repeat(np.array(range(0, len(y) * n_repetitions, len(y))), len(y))
                indexes_ext = np.tile(self.indexes_, (n_repetitions, 1)) + np.transpose(np.tile(adapt_indexs,
                                                                                                (
                                                                                                    self.n_quantifiers,
                                                                                                    1)))
            #  fit quantifiers
            if self.verbose > 0:
                print('Class %s: Training quantifiers...' % self.__class__.__name__, end='')

            self.quantifiers_ = np.zeros(self.n_quantifiers, dtype=object)
            for i in range(self.n_quantifiers):
                self.quantifiers_[i] = deepcopy(self.base_quantifier)
                if predictions_train.ndim == 1:
                    self.quantifiers_[i].fit(X=None, y=y_ext[indexes_ext[:, i]],
                                             predictions_train=predictions_train[indexes_ext[:, i]])
                elif predictions_train.ndim == 2:
                    self.quantifiers_[i].fit(X=None, y=y_ext[indexes_ext[:, i]],
                                             predictions_train=predictions_train[indexes_ext[:, i], :])
                else:
                    #  3 dimensions, an EnsembleOfClassifiers is used
                    self.quantifiers_[i].fit(X=None, y=y_ext[indexes_ext[:, i]],
                                             predictions_train=predictions_train[indexes_ext[:, i], i, :])

            #  compute distributions for distribution_similarity combination strategy
            if n_classes == 2:
                n_descriptors = 1  # number of groups of probabilities used to represent the distribution
            else:
                n_descriptors = n_classes
            # compute bincuts according to bin_strategy
            self.bincuts_ = np.zeros((n_descriptors, self.n_bins + 1))
            for descr in range(n_descriptors):
                if predictions_train.ndim == 2:
                    self.bincuts_[descr, :] = compute_bincuts(x=predictions_train[:, descr], y=y_ext,
                                                              classes=self.classes_,
                                                              n_bins=self.n_bins, bin_strategy=self.bin_strategy)
                elif predictions_train.ndim == 3:
                    self.bincuts_[descr, :] = compute_bincuts(x=predictions_train.mean(axis=1)[:, descr], y=y_ext,
                                                              classes=self.classes_,
                                                              n_bins=self.n_bins, bin_strategy=self.bin_strategy)

            #  compute distributions
            self.distributions_ = np.zeros((self.n_quantifiers, self.n_bins * n_descriptors))
            for i in range(self.n_quantifiers):
                for descr in range(n_descriptors):
                    if predictions_train.ndim == 2:
                        self.distributions_[i, descr * self.n_bins:(descr + 1) * self.n_bins] = \
                             np.histogram(predictions_train[:, descr],
                                          bins=self.bincuts_[descr, :])[0] / len(predictions_train)
                    elif predictions_train.ndim == 3:
                        self.distributions_[i, descr * self.n_bins:(descr + 1) * self.n_bins] = \
                            np.histogram(predictions_train.mean(axis=1)[:, descr],
                                         bins=self.bincuts_[descr, :])[0] / len(predictions_train)
                    # compute cdf if necessary
                    if self.distribution_function == 'CDF':
                        self.distributions_[i, descr * self.n_bins:(descr + 1) * self.n_bins] = \
                            np.cumsum(self.distributions_[i, descr * self.n_bins:(descr + 1) * self.n_bins], axis=1)

        else:
            raise ValueError("Base quantifiers of class %s not implemented in EoQ"
                             % self.base_quantifier.__class__.__name__)

        if self.verbose > 0:
            print('fitted')

        return self

    def predict(self, X, predictions_test=None):
        """

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, (default=None) shape (n_examples, n_classes) if ensemble_estimator_train is not COMPLETE

                Predictions for the testing bag

            Returns
            -------
            prevalences: ndarray, shape(n_classes, ) if an individual combination strategy is selected or
                         a dictionary with the predictions for all strategies if 'all' is selected
                Each value contains the predicted prevalence for the corresponding class
        """
        if isinstance(self.base_quantifier, UsingClassifiers) and (self.ensemble_estimator_test is None and
                                                                   predictions_test is None):
            raise ValueError("ensemble_estimator_test or predictions_test must be not None "
                             "to compute a prediction with objects of class %s" % self.__class__.__name__)

        if self.verbose > 0:
            print('Class %s: Computing predictions for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        prevalences = np.zeros((self.n_quantifiers, n_classes))

        if isinstance(self.base_quantifier, WithoutClassifiers):

            for i in range(self.n_quantifiers):
                prevalences[i, :] = self.quantifiers_[i].predict(X=X)

        elif isinstance(self.base_quantifier, UsingClassifiers):

            if predictions_test is None:
                if self.base_quantifier.probabilistic_predictions:
                    predictions_test = self.ensemble_estimator_test.predict_proba(X)
                else:
                    predictions_test = self.ensemble_estimator_test.predict(X)

            for i in range(self.n_quantifiers):
                if predictions_test.ndim <= 2:
                    prevalences[i, :] = self.quantifiers_[i].predict(X=None, predictions_test=predictions_test)
                else:
                    #  3 dimensions, an EnsembleOfClassifiers is used
                    prevalences[i, :] = self.quantifiers_[i].predict(X=None, predictions_test=predictions_test[:, i, :])

        if self.combination_strategy == 'prevalence_similarity' or self.combination_strategy == 'all':
            p = np.mean(prevalences, axis=0)
            difference = np.zeros(self.n_quantifiers)
            for i in range(self.n_quantifiers):
                difference[i] = mean_absolute_error(p, prevalences[i])
            indxs = np.argsort(difference)
            prevalence_similarity_selected = indxs[0:round(self.n_quantifiers * self.percentage_of_selected_models)]

        if self.combination_strategy == 'distribution_similarity' or self.combination_strategy == 'all':

            if isinstance(self.base_quantifier, WithoutClassifiers):
                #  compute test distribution
                test_distribution = np.zeros((1, self.n_bins * X.shape[1]))
                for att in range(X.shape[1]):
                    test_distribution[0, att * self.n_bins:(att + 1) * self.n_bins] = \
                        np.histogram(X[:, att], bins=self.bincuts_[att, :])[0] / len(X)
                    if self.distribution_function == 'CDF':
                        test_distributions[0, att * self.n_bins:(att + 1) * self.n_bins] = \
                            np.cumsum(test_distribution[0, att * self.n_bins:(att + 1) * self.n_bins], axis=1)
            elif isinstance(self.base_quantifier, UsingClassifiers):

                if predictions_test.ndim == 1:
                    raise ValueError("Class %s: distribution_similarity not implemented with quantifiers that use "
                                     "crisp preditions, used prevalence_similarity instead", self.__class__.__name__)

                if n_classes == 2:
                    n_descriptors = 1  # number of groups of probabilities used to represent the distribution
                else:
                    n_descriptors = n_classes

                #  compute test distributions
                test_distribution = np.zeros((1, self.n_bins * n_descriptors))
                for descr in range(n_descriptors):
                    if predictions_test.ndim == 2:
                        test_distribution[0, descr * self.n_bins:(descr + 1) * self.n_bins] = \
                            np.histogram(predictions_test[:, descr],
                                         bins=self.bincuts_[descr, :])[0] / len(predictions_test)
                    elif predictions_test.ndim == 3:
                        test_distribution[0, descr * self.n_bins:(descr + 1) * self.n_bins] = \
                            np.histogram(predictions_test.mean(axis=1)[:, descr],
                                         bins=self.bincuts_[descr, :])[0] / len(predictions_test)
                    # compute cdf if necessary
                    if self.distribution_function == 'CDF':
                        test_distribution[0, descr * self.n_bins:(descr + 1) * self.n_bins] = \
                            np.cumsum(test_distribution[0, descr * self.n_bins:(descr + 1) * self.n_bins], axis=1)
            else:
                raise ValueError("Base quantifiers of class %s not implemented in EoQ"
                                 % self.base_quantifier.__class__.__name__)
            distances = cdist(test_distribution, self.distributions_, metric=self.distance_bags)
            indxs = np.argsort(distances.squeeze())
            distribution_similarity_selected = indxs[0:round(self.n_quantifiers * self.percentage_of_selected_models)]

        prediction = 0
        if self.combination_strategy == 'mean':
            prediction = np.mean(prevalences, axis=0)
            prediction = prediction / float(np.sum(prediction))
        elif self.combination_strategy == 'median':
            prediction = np.median(prevalences, axis=0)
            prediction = prediction / float(np.sum(prediction))
        elif self.combination_strategy == 'prevalence_similarity':
            prediction = np.mean(prevalences[prevalence_similarity_selected, :], axis=0)
            prediction = prediction / float(np.sum(prediction))
        elif self.combination_strategy == 'distribution_similarity':
            prediction = np.mean(prevalences[distribution_similarity_selected, :], axis=0)
            prediction = prediction / float(np.sum(prediction))
        elif self.combination_strategy == 'all':
            prediction = {}
            m0 = np.mean(prevalences, axis=0)
            m0 = m0 / float(np.sum(m0))
            prediction['mean'] = m0
            m1 = np.median(prevalences, axis=0)
            m1 = m1 / float(np.sum(m1))
            prediction['median'] = m1
            m2 = np.mean(prevalences[prevalence_similarity_selected, :], axis=0)
            m2 = m2 / float(np.sum(m2))
            prediction['prevalence_similarity'] = m2
            m3 = np.mean(prevalences[distribution_similarity_selected, :], axis=0)
            m3 = m3 / float(np.sum(m3))
            prediction['distribution_similarity'] = m3
            prediction['all'] = (m0 + m1 + m2 + m3)/4

        return prediction
