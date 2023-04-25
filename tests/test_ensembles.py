import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from quantificationlib.baselines.ac import PAC
from quantificationlib.multiclass.df import HDX


from quantificationlib.multiclass.em import EM


from quantificationlib.estimators.ensembles import EnsembleOfClassifiers
from quantificationlib.ensembles.eoq import EoQ

from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.multiclass import mean_absolute_error

from quantificationlib.data_utils import load_data, normalize


def test_ensembles():

    dataset = 'examples/datasets/binary/iris.3.csv'
    master_seed = 2032
    n_bags = 5

    method_name = ['EOQ-HDX', 'EOQ-PAC', 'EOQ-EM']

    results = np.zeros((n_bags, len(method_name)))
   
    X, y = load_data(dataset)

    current_seed = master_seed

    # generating training-test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                                                        random_state=current_seed)
    X_train, X_test = normalize(X_train, X_test)

    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they checked whether the estimator is already fitted (by a previous object) or not

    estimator = RandomForestClassifier(n_estimators=100, random_state=master_seed, class_weight='balanced')

    ensemble_estimator = EnsembleOfClassifiers(base_estimator=estimator)

    bag_generator_eoq = PriorShift_BagGenerator(n_bags=100, bag_size=len(X_train),
                                                min_prevalence=0.05, random_state=current_seed)

    #  HDX
    eoq_hdx = EoQ(base_quantifier=HDX(),
                    n_quantifiers=10, bag_generator=bag_generator_eoq,
                    combination_strategy='prevalence_similarity', verbose=1)
    eoq_hdx.fit(X_train, y_train)
    #  PAC
    eoq_pac = EoQ(base_quantifier=PAC(),
                    n_quantifiers=10, bag_generator=bag_generator_eoq,
                    combination_strategy='median',
                    ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator, verbose=1)
    eoq_pac.fit(X_train, y_train)

    # EM
    eoq_em = EoQ(base_quantifier=EM(),
                    n_quantifiers=10, bag_generator=bag_generator_eoq,
                    combination_strategy='all',
                    ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator, verbose=1)
    eoq_em.fit(X_train, y_train)

    #  Testing bags
    bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                            min_prevalence=None, random_state=current_seed)

    prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
    for n_bag in range(n_bags):

        prev_preds = [
            eoq_hdx.predict(X_test[indexes[:, n_bag], :]),
            eoq_pac.predict(X_test[indexes[:, n_bag], :]),
            eoq_em.predict(X_test[indexes[:, n_bag], :])['all'],
        ]

        for n_method, prev_pred in enumerate(prev_preds):
            results[n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

    avg = np.mean(results, axis=0)
    for name, result in zip(method_name,avg):
        assert result >= 0 and result <= 0.05, "Error in method %s " % name