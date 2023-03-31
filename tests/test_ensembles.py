import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LinearRegression

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.baselines.cc import CC, PCC

from quantificationlib.binary.debias import DeBias
from quantificationlib.binary.emd import SORDy
from quantificationlib.binary.quantiles import QUANTy

from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.df import HDX, HDy, MMy, DFX, DFy
from quantificationlib.multiclass.friedman import FriedmanME
from quantificationlib.multiclass.energy import EDX, EDy, CvMy
from quantificationlib.multiclass.regression import REGX, REGy
from quantificationlib.multiclass.knn import PWKQuantifier

from quantificationlib.estimators.ensembles import EnsembleOfClassifiers
from quantificationlib.ensembles.eoq import EoQ

from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.multiclass import hd, l1, l2, topsoe, mean_absolute_error

from quantificationlib.data_utils import load_data, normalize


def test_ensembles():

    dataset = 'examples/datasets/binary/iris.3.csv'
    master_seed = 2032
    n_bags = 5

    method_name = ['EOQ-PCC', 'EOQ-PAC', 'EOQ-EM']

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

    #  PCC
    eoq_pcc = EoQ(base_quantifier=PCC(),
                    n_quantifiers=10, bag_generator=bag_generator_eoq,
                    combination_strategy='mean',
                    ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
    eoq_pcc.fit(X_train, y_train)
    #  PAC
    eoq_pac = EoQ(base_quantifier=PAC(),
                    n_quantifiers=10, bag_generator=bag_generator_eoq,
                    combination_strategy='mean',
                    ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
    eoq_pac.fit(X_train, y_train)

    # EM
    eoq_em = EoQ(base_quantifier=EM(),
                    n_quantifiers=10, bag_generator=bag_generator_eoq,
                    combination_strategy='mean',
                    ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
    eoq_em.fit(X_train, y_train)

    #  Testing bags
    bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                            min_prevalence=None, random_state=current_seed)

    prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
    for n_bag in range(n_bags):

        prev_preds = [
            eoq_pcc.predict(X_test[indexes[:, n_bag], :]),
            eoq_pac.predict(X_test[indexes[:, n_bag], :]),
            eoq_em.predict(X_test[indexes[:, n_bag], :]),
        ]

        for n_method, prev_pred in enumerate(prev_preds):
            results[n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

    avg = np.mean(results, axis=0)
    for name, result in zip(method_name,avg):
        assert result >= 0 and result <= 0.05, "Error in method %s " % name