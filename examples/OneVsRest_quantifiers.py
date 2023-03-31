import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.baselines.cc import CC, PCC

from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.df import HDX, HDy, MMy, DFX, DFy
from quantificationlib.multiclass.friedman import FriedmanME
from quantificationlib.multiclass.energy import EDX, EDy, CvMy
from quantificationlib.multiclass.regression import REGX, REGy
from quantificationlib.multiclass.knn import PWKQuantifier

from quantificationlib.decomposition.multiclass import OneVsRestQuantifier

from quantificationlib.estimators.cross_validation import CV_estimator
from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.multiclass import hd, l1, l2, topsoe, mean_absolute_error

from quantificationlib.data_utils import load_data, normalize


def main(dataset, estimator_name, n_reps, n_bags, master_seed):

    method_name = ['OVR-CC', 'OVR-AC',
                   'OVR-PCC', 'OVR-PAC', 'OVR-EM',
                   'OVR-HDX-8b', 'OVR-CDFX-100b-L1', 'OVR-CDFX-100b-L2',
                   'OVR-HDy-8b', 'OVR-MMy-100', 'OVR-CDFy-100b-L1', 'OVR-CDFy-100b-L2',
                   'OVR-FriedmanME',
                   'OVR-EDX', 'OVR-EDy', 'OVR-CvMy', 'OVR-REGX', 'OVR-REGy',
                   'OVR-PWKQuantifier']

    #  method_name = ['REGX', 'Regy']

    results = np.zeros((n_reps * n_bags, len(method_name)))

    path, fname = os.path.split(dataset)
    print('Dataset: %s' % fname)
    X, y = load_data(dataset)

    for n_rep in range(n_reps):

        current_seed = master_seed + n_rep
        print("*** Training rep {}, seed {}".format(n_rep + 1, current_seed))

        # generating training-test partition
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=current_seed)
        X_train, X_test = normalize(X_train, X_test)

        # classifiers are fitted by each object (all methods will use exactly the same predictions)
        # but they checked whether the estimator is already fitted (by a previous object) or not
        if estimator_name == 'RF':
            estimator = RandomForestClassifier(n_estimators=100, random_state=master_seed, class_weight='balanced')
        else:
            skf_train = StratifiedKFold(n_splits=10, shuffle=True, random_state=master_seed)
            estimator = CV_estimator(estimator=RandomForestClassifier(n_estimators=100, random_state=master_seed,
                                                                      class_weight='balanced'), cv=skf_train)
        ovr_estimator = OneVsRestClassifier(estimator, n_jobs=-1)

        #  AC
        ovr_ac = OneVsRestQuantifier(base_quantifier=AC(),
                                     estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_ac.fit(X_train, y_train)
        #  CC
        ovr_cc = OneVsRestQuantifier(base_quantifier=CC(), estimator_test=ovr_estimator)
        ovr_cc.fit(X_train, y_train)
        #  PCC
        ovr_pcc = OneVsRestQuantifier(base_quantifier=PCC(), estimator_test=ovr_estimator)
        ovr_pcc.fit(X_train, y_train)
        #  PAC
        ovr_pac = OneVsRestQuantifier(base_quantifier=PAC(),
                                      estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_pac.fit(X_train, y_train)

        # EM
        ovr_em = OneVsRestQuantifier(base_quantifier=EM(),
                                     estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_em.fit(X_train, y_train)

        #  HDX
        ovr_hdx = OneVsRestQuantifier(base_quantifier=HDX(n_bins=8))
        ovr_hdx.fit(X_train, y_train)

        #  CDFX-L1
        ovr_cdfx_l1 = OneVsRestQuantifier(base_quantifier=DFX(distribution_function='CDF', n_bins=100, distance='L1'))
        ovr_cdfx_l1.fit(X_train, y_train)
        #  CDFX-L2
        ovr_cdfx_l2 = OneVsRestQuantifier(base_quantifier=DFX(distribution_function='CDF', n_bins=100, distance='L2'))
        ovr_cdfx_l2.fit(X_train, y_train)

        #  HDY
        ovr_hdy = OneVsRestQuantifier(base_quantifier=HDy(n_bins=8),
                                      estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_hdy.fit(X_train, y_train)

        #  MMy
        ovr_mmy = OneVsRestQuantifier(base_quantifier=MMy(n_bins=100),
                                  estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_mmy.fit(X_train, y_train)

        #  CDFy-L1
        ovr_cdfy_l1 = OneVsRestQuantifier(base_quantifier=DFy(distribution_function='CDF', n_bins=100, distance='L1'),
                                          estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_cdfy_l1.fit(X_train, y_train)
        #  CDFy-L2
        ovr_cdfy_l2 = OneVsRestQuantifier(base_quantifier=DFy(distribution_function='CDF', n_bins=100, distance='L2'),
                                          estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_cdfy_l2.fit(X_train, y_train)

        #  FriedmanME - HD
        ovr_me = OneVsRestQuantifier(base_quantifier=FriedmanME(),
                                     estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_me.fit(X_train, y_train)

        #  EDX
        ovr_edx = OneVsRestQuantifier(base_quantifier=EDX())
        ovr_edx.fit(X_train, y_train)
        #  EDy
        ovr_edy = OneVsRestQuantifier(base_quantifier=EDy(),
                                      estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_edy.fit(X_train, y_train)
        #  CvMy
        ovr_cvmy = OneVsRestQuantifier(base_quantifier=CvMy(),
                                       estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_cvmy.fit(X_train, y_train)

        #  REGX
        bag_generator = PriorShift_BagGenerator(n_bags=1000, bag_size=len(X_train),
                                                min_prevalence=None, random_state=master_seed)
        ovr_regx = OneVsRestQuantifier(base_quantifier=REGX(bag_generator=bag_generator, n_bins=8,
                                                            bin_strategy='equal_width', regression_estimator=None))
        ovr_regx.fit(X_train, y_train)
        # REGy
        ovr_regy = OneVsRestQuantifier(base_quantifier=REGy(bag_generator=bag_generator, n_bins=8, bin_strategy='equal_width',
                                                            regression_estimator=None),
                                       estimator_train=ovr_estimator, estimator_test=ovr_estimator)
        ovr_regy.fit(X_train, y_train)

        #  PWK
        ovr_knn_q = OneVsRestQuantifier(base_quantifier=PWKQuantifier())
        ovr_knn_q.fit(X_train, y_train)

        #  Testing bags
        bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                                min_prevalence=None, random_state=current_seed)

        prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
        for n_bag in range(n_bags):

            prev_preds = [
                ovr_cc.predict(X_test[indexes[:, n_bag], :]),
                ovr_ac.predict(X_test[indexes[:, n_bag], :]),
                ovr_pcc.predict(X_test[indexes[:, n_bag], :]),
                ovr_pac.predict(X_test[indexes[:, n_bag], :]),
                ovr_em.predict(X_test[indexes[:, n_bag], :]),
                ovr_hdx.predict(X_test[indexes[:, n_bag], :]),
                ovr_cdfx_l1.predict(X_test[indexes[:, n_bag], :]),
                ovr_cdfx_l2.predict(X_test[indexes[:, n_bag], :]),
                ovr_hdy.predict(X_test[indexes[:, n_bag], :]),
                ovr_mmy.predict(X_test[indexes[:, n_bag], :]),
                ovr_cdfy_l1.predict(X_test[indexes[:, n_bag], :]),
                ovr_cdfy_l2.predict(X_test[indexes[:, n_bag], :]),
                ovr_me.predict(X_test[indexes[:, n_bag], :]),
                ovr_edx.predict(X_test[indexes[:, n_bag], :]),
                ovr_edy.predict(X_test[indexes[:, n_bag], :]),
                ovr_cvmy.predict(X_test[indexes[:, n_bag], :]),
                ovr_regx.predict(X_test[indexes[:, n_bag], :]),
                ovr_regy.predict(X_test[indexes[:, n_bag], :]),
                ovr_knn_q.predict(X_test[indexes[:, n_bag], :])
            ]

            for n_method, prev_pred in enumerate(prev_preds):
                results[n_rep * n_bags + n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

    #  printing and saving results
    filename = 'results/OneVsRest-quantifiers-' + estimator_name
    #  all
    np.savetxt(filename + '-all-' + str(n_reps) + '-' + str(n_bags), results,
               fmt='%.5f', delimiter=",", header=','.join(method_name))
    #  avg
    file_avg = open(filename + '-avg-' + str(n_reps) + '-' + str(n_bags), 'w')
    avg = np.mean(results, axis=0)
    print('\nMAE results')
    print('-' * 26)
    for n_method, method in enumerate(method_name):
        file_avg.write('%-19s%.5f\n' % (method, avg[n_method]))
        print('%-19s%.5f' % (method, avg[n_method]))


if __name__ == '__main__':
    main(dataset='datasets/multiclass/iris.csv', estimator_name='CV',
         n_reps=2, n_bags=50, master_seed=2032)

# MAE results
# --------------------------
# OVR-CC             0.02754
# OVR-AC             0.03825
# OVR-PCC            0.02520
# OVR-PAC            0.02444
# OVR-EM             0.01982
# OVR-HDX-8b         0.04279
# OVR-CDFX-100b-L1   0.13427
# OVR-CDFX-100b-L2   0.11144
# OVR-HDy-8b         0.02562
# OVR-MMy-100        0.02833
# OVR-CDFy-100b-L1   0.02833
# OVR-CDFy-100b-L2   0.02487
# OVR-FriedmanME     0.03156
# OVR-EDX            0.11215
# OVR-EDy            0.02489
# OVR-CvMy           0.07851
# OVR-REGX           0.03497
# OVR-REGy           0.03239
# OVR-PWKQuantifier  0.03046