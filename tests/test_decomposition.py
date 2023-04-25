import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from quantificationlib.baselines.ac import AC
from quantificationlib.multiclass.df import HDX

from quantificationlib.decomposition.multiclass import OneVsRestQuantifier

from quantificationlib.metrics.ordinal import emd, emd_score

from quantificationlib.estimators.frank_and_hall import FrankAndHallClassifier
from quantificationlib.estimators.frank_and_hall import FrankAndHallMonotoneClassifier
from quantificationlib.decomposition.ordinal import FrankAndHallQuantifier

from quantificationlib.estimators.cross_validation import CV_estimator
from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.multiclass import mean_absolute_error

from quantificationlib.data_utils import load_data, normalize


def test_decomposition_multiclass():

    dataset='examples/datasets/multiclass/iris.csv'
    n_bags=5
    master_seed=2032

    method_name = ['OVR-HDX', 'OVR-AC']

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
    ovr_hdx = OneVsRestQuantifier(base_quantifier=HDX())
    ovr_hdx.fit(X_train, y_train)
    

    #  Testing bags
    bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                            min_prevalence=None, random_state=current_seed)

    prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
    for n_bag in range(n_bags):

        prev_preds = [
            ovr_hdx.predict(X_test[indexes[:, n_bag], :]),
            ovr_ac.predict(X_test[indexes[:, n_bag], :]),
        ]

        for n_method, prev_pred in enumerate(prev_preds):
            results[n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

    #  printing and saving results
    avg = np.mean(results, axis=0)
    for name, result in zip(method_name,avg):
        assert result >= 0 and result <= 0.026, "Error in method %s " % name

def test_decomposition_ordinal():

    dataset='examples/datasets/ordinal/ESL.csv'
    n_bags=5
    master_seed=2032

    method_name = ['FH-CC', 'FH-AC', 'FHM-AC']

    results_emd = np.zeros((n_bags, len(method_name)))
    results_emd_score = np.zeros((n_bags, len(method_name)))

    X, y = load_data(dataset)

    current_seed = master_seed

    # generating training-test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=current_seed)
    X_train, X_test = normalize(X_train, X_test)

    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they checked whether the estimator is already fitted (by a previous object) or not
    fh = FrankAndHallClassifier(estimator=RandomForestClassifier(n_estimators=100,
                                                                    random_state=master_seed,
                                                                    class_weight='balanced'), n_jobs=-1, verbose=1)
    fhm = FrankAndHallMonotoneClassifier(estimator=RandomForestClassifier(n_estimators=100,
                                                                    random_state=master_seed,
                                                                    class_weight='balanced'), n_jobs=-1, verbose=1)


    #  AC
    fh_ac = FrankAndHallQuantifier(quantifier=AC(), estimator_train=fh, estimator_test=fh, verbose=1)
    fh_ac.fit(X_train, y_train)
    #  AC
    fhm_ac = FrankAndHallQuantifier(quantifier=AC(), estimator_train=fh, estimator_test=fhm, verbose=1)
    fhm_ac.fit(X_train, y_train)
    #  CC
    fh_hdx = FrankAndHallQuantifier(quantifier=HDX(), estimator_train=None, estimator_test=None, verbose=1)
    fh_hdx.fit(X_train, y_train)
    

    bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                            min_prevalence=None, random_state=current_seed)

    prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
    for n_bag in range(n_bags):
        prev_preds = [
            fh_hdx.predict(X_test[indexes[:, n_bag], :]),
            fh_ac.predict(X_test[indexes[:, n_bag], :]),
            fhm_ac.predict(X_test[indexes[:, n_bag], :]),
        ]

    for n_method, prev_pred in enumerate(prev_preds):
        results_emd[n_bag, n_method] = emd(prev_true[:, n_bag], prev_pred)
        results_emd_score[n_bag, n_method] = emd_score(prev_true[:, n_bag], prev_pred)

    avg_emd = np.mean(results_emd, axis=0)
    avg_emd_score = np.mean(results_emd_score, axis=0)
    for name, result in zip(method_name,avg_emd):
        assert result >= 0 and result <= 0.05, "Error in method %s " % name

    for name, result in zip(method_name,avg_emd_score):
        assert result >= 0 and result <= 0.2, "Error in method %s " % name