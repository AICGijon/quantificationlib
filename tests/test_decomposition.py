import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from quantificationlib.baselines.ac import AC
from quantificationlib.baselines.cc import CC

from quantificationlib.decomposition.multiclass import OneVsRestQuantifier

from quantificationlib.estimators.cross_validation import CV_estimator
from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.multiclass import mean_absolute_error

from quantificationlib.data_utils import load_data, normalize


def test_decomposition_multiclass():

    dataset='examples/datasets/multiclass/iris.csv'
    n_bags=5
    master_seed=2032

    method_name = ['OVR-CC', 'OVR-AC']

    results = np.zeros((n_bags, len(method_name)))

    X, y = load_data(dataset)


    current_seed = master_seed

    # generating training-test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=current_seed)
    X_train, X_test = normalize(X_train, X_test)

    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they checked whether the estimator is already fitted (by a previous object) or not
    estimator = RandomForestClassifier(n_estimators=100, random_state=master_seed, class_weight='balanced')
        
    ovr_estimator = OneVsRestClassifier(estimator, n_jobs=-1)

    #  AC
    ovr_ac = OneVsRestQuantifier(base_quantifier=AC(),
                                    estimator_train=ovr_estimator, estimator_test=ovr_estimator)
    ovr_ac.fit(X_train, y_train)
    #  CC
    ovr_cc = OneVsRestQuantifier(base_quantifier=CC(), estimator_test=ovr_estimator)
    ovr_cc.fit(X_train, y_train)
    

    #  Testing bags
    bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                            min_prevalence=None, random_state=current_seed)

    prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
    for n_bag in range(n_bags):

        prev_preds = [
            ovr_cc.predict(X_test[indexes[:, n_bag], :]),
            ovr_ac.predict(X_test[indexes[:, n_bag], :]),
        ]

        for n_method, prev_pred in enumerate(prev_preds):
            results[n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

    #  printing and saving results
    avg = np.mean(results, axis=0)
    for name, result in zip(method_name,avg):
        assert result >= 0 and result <= 0.02, "Error in method %s " % name


import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from quantificationlib.baselines.ac import AC
from quantificationlib.baselines.cc import CC

from quantificationlib.estimators.frank_and_hall import FrankAndHallClassifier
from quantificationlib.decomposition.ordinal import FrankAndHallQuantifier

from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.ordinal import emd

from quantificationlib.data_utils import load_data, normalize


def test_decomposition_ordinal():

    dataset='examples/datasets/ordinal/ESL.csv'
    n_bags=5
    master_seed=2032

    method_name = ['FH-CC', 'FH-AC']

    results = np.zeros((n_bags, len(method_name)))

    X, y = load_data(dataset)

    current_seed = master_seed

    # generating training-test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=current_seed)
    X_train, X_test = normalize(X_train, X_test)

    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they checked whether the estimator is already fitted (by a previous object) or not
    fh = FrankAndHallClassifier(estimator=RandomForestClassifier(n_estimators=100,
                                                                    random_state=master_seed,
                                                                    class_weight='balanced'), n_jobs=-1)

    #  AC_HD
    fh_ac = FrankAndHallQuantifier(quantifier=AC(), estimator_train=fh, estimator_test=fh)
    fh_ac.fit(X_train, y_train)
    #  CC
    fh_cc = FrankAndHallQuantifier(quantifier=CC(), estimator_train=None, estimator_test=fh)
    fh_cc.fit(X_train, y_train)
    

    bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                            min_prevalence=None, random_state=current_seed)

    prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
    for n_bag in range(n_bags):
        prev_preds = [
            fh_cc.predict(X_test[indexes[:, n_bag], :]),
            fh_ac.predict(X_test[indexes[:, n_bag], :]),
        ]

    for n_method, prev_pred in enumerate(prev_preds):
        results[n_bag, n_method] = emd(prev_true[:, n_bag], prev_pred)

    avg = np.mean(results, axis=0)
    for name, result in zip(method_name,avg):
        assert result >= 0 and result <= 0.05, "Error in method %s " % name