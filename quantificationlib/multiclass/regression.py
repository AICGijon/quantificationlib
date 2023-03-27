import numpy as np
import statsmodels.api as sm
import six

from abc import ABCMeta

from copy import deepcopy

from quantificationlib.base import UsingClassifiers, WithoutClassifiers
from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.multiclass.df import compute_bincuts

from sklearn.linear_model import LinearRegression


class REG(six.with_metaclass(ABCMeta)):
    """ REG base class for REGX y REGy

        The idea of these quantifiers is to learn a regression model able to predict the prevalences. To learn said
        regression model, this kind of objects generates a training set of bag of examples using a selected kind of
        shift (prior probability shift, covariate shift or a mix of both). The training set contains a collection of
        pairs (PDF distribution, prevalences) in which each pair is obtained from a bag of examples. The PDF tries
        to capture the distribution of the bag.

        Parameters
        ----------
        bag_generator : BagGenerator object (default=PriorShift_BagGenerator())
            Object to generate the bags with a selected shift

        n_bins : int (default=8)
            Number of bins to compute the PDF of each distribution

        bin_strategy : str (default='normal')
            Method to compute the boundaries of the bins:
                'equal_width': bins of equal length (it could be affected by outliers)
                'equal_count': bins of equal counts (considering the examples of all classes)
                'binormal': (Only for binary quantification) It is inspired on the method devised by
                            (Tasche, 2019, Eq (A16b)). the cut points, $-\infty < c_1 < \ldots < c_{b-1} < \infty$,
                            are computed as follows based on the assumption that the features follow a normal
                            distribution:

                          $c_i = \frac{\sigma^+ + \sigma^{-}}{2} \ \Phi^{-1}\bigg(\frac{i}{b}\bigg)  + \frac{\mu^+ + \mu^{-}}{2} ,  \quad i=1,\ldots,b-1$

                            where $\Phi^{-1}$ is the quantile function of the standard normal distribution, and $\mu$
                            and $\sigma$ of the normal distribution are estimated as the average of those values for
                            the training examples of each class.

                'normal':  The idea is that each feacture follows a normal distribution. $\mu$ and $\sigma$ are
                           estimated as the weighted mean and std from the training distribution. The cut points
                           $-\infty < c_1 < \ldots < c_{b-1} < \infty$ are computed as follows:

                           $c_i = \sigma^ \ \Phi^{-1}\bigg(\frac{i}{b}\bigg)  + \mu ,  \quad i=1,\ldots,b-1$

        regression_estimator: estimator object (default=None)
            A regression estimator object. If the value is None the regression estimator used is a Generalized Linear
            Model (GLM) from statsmodels package with logit link and Binomial family as parameters (see Baum 2008).
            It is used to learn a regression model able to predict the prevalence for each class, so the method will
            fit as many regression estimators as classes in multiclass problems and just one for binary problems.

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        bag_generator : BagGenerator object
            Object to generate the bags with a selected shift

        n_bins : int
            Number of bins to compute the PDF of each distribution

        bin_strategy : str
            Method to compute the boundaries of the bins

        regression_estimator: estimator object, None
            A regression estimator object

        verbose : int
            The verbosity level

        dataX_ : array-like, shape(n_bags, n_features)
            X data for training REGX/REGy's regressor model. Each row corresponds to the collection of histograms (one
            per input feature) of the corresponding bag

        dataY_ : array-like, shape(n_bags, n_classes)
            Y data for training REGX/REGy's regressor model. Each value corresponds to the prevalences of the
            corresponding bag

        bincuts_ : ndarray, shape (n_features, n_bins + 1)
            Bin cuts for each feature

        estimators_ : array of estimators, shape (n_classes, ) multiclass (1, ) binary quantification
            It stores the estimators. For multiclass problems, the method learns an individual estimator
            for each class

        models_ : array of models, i.e., fitted estimators, shape (n_classes, )
            This is the fitted regressor model for each class. It is needed when regression_estimator is None and
            a GML models are used (these objects do not store the fitted model).

        n_classes_ : int
            The number of classes

        References
        ----------
        Christopher F. Baum: Stata tip 63: Modeling proportions. The Stata Journal 8.2 (2008): 299-303
    """

    def __init__(self, bag_generator=PriorShift_BagGenerator(), n_bins=8, bin_strategy='equal_width',
                 regression_estimator=None, verbose=0, **kwargs):
        super(REG, self).__init__(**kwargs)
        self.bag_generator = bag_generator
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.regression_estimator = regression_estimator
        self.verbose = verbose
        self.dataX_ = None
        self.dataY_ = None
        self.bincuts_ = None
        self.estimators_ = None
        self.models_ = None
        self.n_classes_ = None

    def create_training_set_of_distributions(self, X, y, att_range=None):
        """ Create a training set for REG objects. Each example corresponds to a histogram of a bag
            of examples generated from (X, y). The size of the complete histogram is n_features * n_bins, because
            it is formed by concatenating the histogram for each input feature. This method computes the values
            for dataX_, dataY_ and bincuts_ attributes

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            att_range: array-like, (2,1)
                Min and Max possible values of the input feature x. These values might not coincide with the actual
                Min and Max values of vector x. For instance, in the case of x represents a set of probabilistic
                predictions, these values will be 0 and 1. These values may be needed by compute_bincuts function
            """

        classes = np.unique(y)
        self.n_classes_ = len(classes)
        n_examples = X.shape[0]
        if len(X.shape) == 1:
            n_features = 1
        else:
            n_features = X.shape[1]

        #  compute bincuts
        self.bincuts_ = np.zeros((n_features, self.n_bins + 1))
        for att in range(n_features):
            self.bincuts_[att, :] = compute_bincuts(x=X[:, att], y=y, classes=classes,
                                                    n_bins=self.n_bins, bin_strategy=self.bin_strategy,
                                                    att_range=att_range)

        #  compute bags
        prevalences, indexes = self.bag_generator.generate_bags(X, y)

        self.dataX_ = np.zeros((self.bag_generator.n_bags, self.n_bins * n_features))
        self.dataY_ = np.transpose(prevalences)

        for nbag in range(self.bag_generator.n_bags):
            for att in range(n_features):
                self.dataX_[nbag, att * self.n_bins:(att + 1) * self.n_bins] = \
                    np.histogram(X[indexes[:, nbag], att], bins=self.bincuts_[att, :])[0] / n_examples

    def fit_regressor(self):
        """ This method trains the regressor model using dataX_ and dataY_
         """
        #  checking if it is a binary problem
        if self.n_classes_ == 2:
            self.dataY_ = self.dataY_[:, 1]
            if self.regression_estimator is not None:
                self.estimators_ = self.regression_estimator.fit(self.dataX_, self.dataY_)
            else:
                self.estimators_ = sm.GLM(endog=self.dataY_, exog=self.dataX_, family=sm.families.Binomial())
                self.models_ = self.estimators_.fit()
        else:
            self.estimators_ = np.zeros(self.n_classes_, dtype=object)
            if self.regression_estimator is not None:
                for n_cls in range(self.n_classes_):
                    self.estimators_[n_cls] = deepcopy(self.regression_estimator)
                    self.estimators_[n_cls].fit(self.dataX_, self.dataY_[:, n_cls])
            else:
                self.models_ = np.zeros(self.n_classes_, dtype=object)
                for n_cls in range(self.n_classes_):
                    self.estimators_[n_cls] = sm.GLM(endog=self.dataY_[:, n_cls], exog=self.dataX_,
                                                     family=sm.families.Binomial())
                    #_[:, self.n_bins * n_cls:self.n_bins * (n_cls + 1)],

                    self.models_[n_cls] = self.estimators_[n_cls].fit()

        return self

    def predict_bag(self, bagX):
        """ This method makes a prediction for a testing bag represented by its PDF, bagX parameter

            Parameters
            ----------
            bagX : array-like, shape (n_bins * n_classes, ) for REGy and (n_bins * n_features, ) for REGX
                Testing bag's PDF

            Returns
            -------
            prevalences: ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        if self.n_classes_ == 2:
            # binary case
            if self.regression_estimator is None:
                p = self.estimators_.predict(params=self.models_.params, exog=bagX)
            else:
                p = self.estimators_.predict(bagX.reshape(1, -1))

            prevalences = [1 - p, p]
            prevalences = np.clip(prevalences, 0, 1)

            if np.sum(prevalences) > 0:
                prevalences = prevalences / float(np.sum(prevalences))
            else:
                raise ValueError("predicted prevalences by object of class %s are all 0", self.__class__.__name__)
            return prevalences.squeeze()

        else:
            #  multiclass
            prevalences = np.zeros(self.n_classes_)
            for n_cls in range(self.n_classes_):
                if isinstance(self.estimators_[n_cls], sm.GLM):
                    prevalences[n_cls] = self.estimators_[n_cls].predict(params=self.models_[n_cls].params, exog=bagX)
                else:
                    prevalences[n_cls] = self.estimators_[n_cls].predict(bagX.reshape(1, -1))

            prevalences = np.clip(prevalences, 0, 1)
            prevalences = prevalences / float(np.sum(prevalences))
            return prevalences.reshape(-1)


class REGX(WithoutClassifiers, REG):
    """ REGX

        The idea is to learn a regression model able to predict the prevalences given a PDF distribution. In this case,
        the distributions are represented using PDFs of the input features (X). To learn such regression model, this
        object generates a training set of bags of examples using a selected kind of shift (prior probability shift,
        covariate shift or a mix of both)

        Parameters
        ----------
        bag_generator : BagGenerator object (default=PriorShift_BagGenerator())
            Object to generate the bags with a selected shift

        n_bins : int (default=8)
            Number of bins to compute the PDF of each distribution

        bin_strategy : str (default='normal')
            Method to compute the boundaries of the bins:
                'equal_width': bins of equal length (it could be affected by outliers)
                'equal_count': bins of equal counts (considering the examples of all classes)
                'binormal': (Only for binary quantification) It is inspired on the method devised by
                            (Tasche, 2019, Eq (A16b)). the cut points, $-\infty < c_1 < \ldots < c_{b-1} < \infty$,
                            are computed as follows based on the assumption that the features follow a normal
                            distribution:

                          $c_i = \frac{\sigma^+ + \sigma^{-}}{2} \ \Phi^{-1}\bigg(\frac{i}{b}\bigg)  + \frac{\mu^+ + \mu^{-}}{2} ,  \quad i=1,\ldots,b-1$

                            where $\Phi^{-1}$ is the quantile function of the standard normal distribution, and $\mu$
                            and $\sigma$ of the normal distribution are estimated as the average of those values for
                            the training examples of each class.

                'normal':  The idea is that each feacture follows a normal distribution. $\mu$ and $\sigma$ are
                           estimated as the weighted mean and std from the training distribution. The cut points
                           $-\infty < c_1 < \ldots < c_{b-1} < \infty$ are computed as follows:

                           $c_i = \sigma^ \ \Phi^{-1}\bigg(\frac{i}{b}\bigg)  + \mu ,  \quad i=1,\ldots,b-1$

        regression_estimator: estimator object (default=None)
            A regression estimator object. If the value is None the regression estimator used is a Generalized Linear
            Model (GLM) from statsmodels package with logit link and Binomial family as parameters (see Baum 2008).
            It is used to learn a regression model able to predict the prevalence for each class, so the method will
            fit as many regression estimators as classes in multiclass problems and just one for binary problems.

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        bag_generator : BagGenerator object
            Object to generate the bags with a selected shift

        n_bins : int
            Number of bins to compute the PDF of each distribution

        bin_strategy : str
            Method to compute the boundaries of the bins

        regression_estimator: estimator object, None
            A regression estimator object

        verbose : int
            The verbosity level

        dataX_ : array-like, shape(n_bags, n_features)
            X data for training REGX's regressor model. Each row corresponds to the collection of histograms (one
            per input feature) of the corresponding bag

        dataY_ : array-like, shape(n_bags, n_classes)
            Y data for training REGX's regressor model. Each value corresponds to the prevalences of the
            corresponding bag

        bincuts_ : ndarray, shape (n_features, n_bins + 1)
            Bin cuts for each feature

        estimators_ : array of estimators, shape (n_classes, ) multiclass (1, ) binary quantification
            It stores the estimators. For multiclass problems, the method learns an individual estimator
            for each class

        models_ : array of models, i.e., fitted estimators, shape (n_classes, )
            This is the fitted regressor model for each class. It is needed when regression_estimator is None and
            a GML models are used (this objects do not store the fitted model).

        n_classes_ : int
            The number of classes

        References
        ----------
        Christopher F. Baum: Stata tip 63: Modeling proportions. The Stata Journal 8.2 (2008): 299-303
    """
    def __init__(self, bag_generator=PriorShift_BagGenerator(), n_bins=8, bin_strategy='normal',
                 regression_estimator=None, verbose=False):
        super(REGX, self).__init__(bag_generator=bag_generator, n_bins=n_bins, bin_strategy=bin_strategy,
                                   regression_estimator=regression_estimator, verbose=verbose)

    def fit(self, X, y):
        """ This method just has two steps: 1) it computes a training dataset formed by a collection of bags of
            examples (using create_training_set_of_distributions) and 2) it trains a regression model using said
            training set just calling fit_regressor, a inherited method from REG base class

             Parameters
             ----------
             X : array-like, shape (n_examples, n_features)
                 Data

             y : array-like, shape (n_examples, )
                 True classes
         """
        if self.verbose > 0:
            print('Class %s: Creating training distribution...' % self.__class__.__name__, end='')

        super(REGX, self).create_training_set_of_distributions(X=X, y=y, att_range=None)

        if self.verbose > 0:
            print('done')

        if self.verbose > 0:
            print('Class %s: Learning regression model...' % self.__class__.__name__, end='')

        super(REGX, self).fit_regressor()

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X):
        """ This method computes the histogram for the testing set X, using the bincuts for each input feature
            computed by fit method and then it makes a prediction applying the regression model using the
            inherited method predict_bag

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            Returns
            -------
            prevalences: ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        if self.verbose > 0:
            print('Class %s: Estimating prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_examples = X.shape[0]
        if len(X.shape) == 1:
            n_features = 1
        else:
            n_features = X.shape[1]

        bagX = np.zeros(n_features * self.n_bins)

        for att in range(n_features):
            bagX[att * self.n_bins:(att + 1) * self.n_bins] = np.histogram(X[:, att],
                                                                           bins=self.bincuts_[att, :])[0] /\
                                                                           n_examples

        return super(REGX, self).predict_bag(bagX)


class REGy(UsingClassifiers, REG):
    """ REGy

        The idea is to learn a regression model able to predict the prevalences given a PDF distribution. In this case,
        the distributions are represented using PDFs of the predictions (y) from a classifer. To learn such regression
        model, this object first trains a classifier using all data and then generates a training set of bags of
        examples (in this case the predictions of each example) using a selected kind of shift (prior probability
        shift, covariate shift or a mix of both)

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to train a classifier using the
            examples of the training set. This classifier is used to obtain the predictions for the training
            examples and to compute the PDF of each class individually using such predictions

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

        bag_generator : BagGenerator object (default=PriorShift_BagGenerator())
            Object to generate the bags with a selected shift

        n_bins : int (default=8)
            Number of bins to compute the PDF of each distribution

        bin_strategy : str (default='normal')
            Method to compute the boundaries of the bins:
                'equal_width': bins of equal length (it could be affected by outliers)
                'equal_count': bins of equal counts (considering the examples of all classes)
                'binormal': (Only for binary quantification) It is inspired on the method devised by
                            (Tasche, 2019, Eq (A16b)). the cut points, $-\infty < c_1 < \ldots < c_{b-1} < \infty$,
                            are computed as follows based on the assumption that the features follow a normal
                            distribution:

                          $c_i = \frac{\sigma^+ + \sigma^{-}}{2} \ \Phi^{-1}\bigg(\frac{i}{b}\bigg)  + \frac{\mu^+ + \mu^{-}}{2} ,  \quad i=1,\ldots,b-1$

                            where $\Phi^{-1}$ is the quantile function of the standard normal distribution, and $\mu$
                            and $\sigma$ of the normal distribution are estimated as the average of those values for
                            the training examples of each class.

                'normal':  The idea is that each feacture follows a normal distribution. $\mu$ and $\sigma$ are
                           estimated as the weighted mean and std from the training distribution. The cut points
                           $-\infty < c_1 < \ldots < c_{b-1} < \infty$ are computed as follows:

                           $c_i = \sigma^ \ \Phi^{-1}\bigg(\frac{i}{b}\bigg)  + \mu ,  \quad i=1,\ldots,b-1$

        regression_estimator: estimator object (default=None)
            A regression estimator object. If it is None the regression estimator used is a Generalized Linear
            Model (GLM) from statsmodels package with logit link and Binomial family as parameters.
            It is used to learn a regression model able to predict the prevalence for each class, so the method will
            fit as many regression estimators as classes in multiclass problem and just one for binary problems.

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        bag_generator : BagGenerator object
            Object to generate the bags with a selected shift

        needs_predictions_train : bool, True
            It is True because PDFy quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
             This means that predictions_train_/predictions_test_ contain probabilistic predictions

        n_bins : int
            Number of bins to compute the PDF of each distribution

        bin_strategy : str
            Method to compute the boundaries of the bins

        regression_estimator: estimator object, None
            A regression estimator object

        verbose : int
            The verbosity level

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the testing bag

        classes_ : ndarray, shape (n_classes, )
            Class labels

        dataX_ : array-like, shape(n_bags, n_features)
            X data for training REGy's regressor model. Each row corresponds to the predictions histogram for the
            examples of the corresponding bag

        dataY_ : array-like, shape(n_bags, n_classes)
            Y data for training REGy's regressor model. Each value corresponds to the prevalences of the
            corresponding bag

        bincuts_ : ndarray, shape (n_features, n_bins + 1)
            Bin cuts for each feature

        estimators_ : array of estimators, shape (n_classes, ) multiclass (1, ) binary quantification
            It stores the estimators. For multiclass problems, the method learns an individual estimator
            for each class

        models_ : array of models, i.e., fitted estimators, shape (n_classes, )
            This is the fitted regressor model for each class. It is needed when regression_estimator is None and
            a GML models are used (this objects do not store the fitted model).

        n_classes_ : int
            The number of classes

        References
        ----------
        Christopher F. Baum: Stata tip 63: Modeling proportions. The Stata Journal 8.2 (2008): 299-303
    """

    def __init__(self, estimator_train=None, estimator_test=None, bag_generator=PriorShift_BagGenerator(),
                 n_bins=8, bin_strategy='equal_width', regression_estimator=None, verbose=False):
        super(REGy, self).__init__(bag_generator=bag_generator, n_bins=n_bins, bin_strategy=bin_strategy,
                                   regression_estimator=regression_estimator, verbose=verbose,
                                   estimator_train=estimator_train, estimator_test=estimator_test,
                                   needs_predictions_train=True, probabilistic_predictions=True)

    def fit(self, X, y, predictions_train=None):
        """ This method just has two steps: 1) it computes a training dataset formed by a collection of bags of
            examples (using create_training_set_of_distributions) and 2) it trains a regression model using said
            training set just calling fit_regressor, a inherited method from REG base class

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
            print('Class %s: Creating training distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        if n_classes == 2:
            preds = self.predictions_train_[:, 0].reshape(-1, 1)
        else:
            preds = self.predictions_train_

        super(REGy, self).create_training_set_of_distributions(X=preds, y=self.y_ext_, att_range=[0, 1])

        if self.verbose > 0:
            print('done')

        if self.verbose > 0:
            print('Class %s: Learning regression model...' % self.__class__.__name__, end='')

        super(REGy, self).fit_regressor()

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ This method first computes the histogram for the testing set X, using the bincuts computed by the fit
            method and the predictions for the testing bag (X, y). These predictions may be explicited given in the
            predictions_test parameter. Then it makes a prediction applying the regression model using the
            inherited method predict_bag
            
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
            prevalences: ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Estimating prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_examples = len(self.predictions_test_)
        n_classes = len(self.classes_)
        if n_classes == 2:
            n_descriptors = 1
        else:
            n_descriptors = n_classes

        bagX = np.zeros(n_descriptors * self.n_bins)

        for descr in range(n_descriptors):
            bagX[descr * self.n_bins:(descr + 1) * self.n_bins] = np.histogram(self.predictions_test_[:, descr],
                                                                               bins=self.bincuts_[descr, :])[0] / \
                                                                               n_examples

        return super(REGy, self).predict_bag(bagX)
