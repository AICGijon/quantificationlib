import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestClassifier

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.baselines.cc import CC, PCC

from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.df import HDX, HDy, DFX, DFy
from quantificationlib.multiclass.friedman import FriedmanME
from quantificationlib.multiclass.energy import EDX, EDy, CvMy
from quantificationlib.multiclass.knn import PWKQuantifier

from quantificationlib.estimators.frank_and_hall import FrankAndHallClassifier
from quantificationlib.decomposition.ordinal import FrankAndHallQuantifier

from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.ordinal import emd, emd_distances

from data_utils import load_data, normalize


def main(dataset, n_reps, n_bags, master_seed):
    method_name = ['FH-CC', 'FH-AC', 'FH-PCC', 'FH-PAC', 'FH-EM',
                   'FH-HDX-8b', 'FH-CDFX-100b-L1',
                   'FH-HDy-8b', 'FH-CDFy-100b-L1',
                   'FH-FriedmanME-L1',
                   'FH-EDX', 'FH-EDy', 'FH-CvMy',
                   'FH-PWKQuantifier']

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
        fh = FrankAndHallClassifier(estimator=RandomForestClassifier(n_estimators=100,
                                                                     random_state=master_seed,
                                                                     class_weight='balanced'), n_jobs=-1)

        #  AC_HD
        fh_ac = FrankAndHallQuantifier(quantifier=AC(), estimator_train=fh, estimator_test=fh)
        fh_ac.fit(X_train, y_train)
        #  CC
        fh_cc = FrankAndHallQuantifier(quantifier=CC(), estimator_train=None, estimator_test=fh)
        fh_cc.fit(X_train, y_train)
        #  PCC
        fh_pcc = FrankAndHallQuantifier(quantifier=PCC(), estimator_train=None, estimator_test=fh)
        fh_pcc.fit(X_train, y_train)
        #  PAC_HD
        fh_pac = FrankAndHallQuantifier(quantifier=PAC(), estimator_train=fh, estimator_test=fh)
        fh_pac.fit(X_train, y_train)

        # EM
        fh_em = FrankAndHallQuantifier(quantifier=EM(), estimator_train=fh, estimator_test=fh)
        fh_em.fit(X_train, y_train)

        #  HDX
        fh_hdx = FrankAndHallQuantifier(quantifier=HDX(n_bins=8))
        fh_hdx.fit(X_train, y_train)

        #  CDFX-L1
        fh_cdfx_l1 = FrankAndHallQuantifier(quantifier=DFX(distribution_function='CDF', n_bins=100, distance='L1')
                                            , estimator_train=fh, estimator_test=fh)
        fh_cdfx_l1.fit(X_train, y_train)

        #  HDY
        fh_hdy = FrankAndHallQuantifier(quantifier=HDy(n_bins=8), estimator_train=fh, estimator_test=fh)
        fh_hdy.fit(X_train, y_train)

        #  CDFy-L1
        fh_cdfy_l1 = FrankAndHallQuantifier(quantifier=DFy(distribution_function='CDF', n_bins=100, distance='L1'),
                                            estimator_train=fh, estimator_test=fh)
        fh_cdfy_l1.fit(X_train, y_train)

        #  FriedmanME - HD
        fh_me = FrankAndHallQuantifier(quantifier=FriedmanME(distance='L1'), estimator_train=fh, estimator_test=fh)
        fh_me.fit(X_train, y_train)

        #  EDX
        fh_edx = FrankAndHallQuantifier(quantifier=EDX())
        fh_edx.fit(X_train, y_train)
        #  EDy
        fh_edy = FrankAndHallQuantifier(quantifier=EDy(), estimator_train=fh, estimator_test=fh)
        fh_edy.fit(X_train, y_train)
        #  CvMy
        fh_cvmy = FrankAndHallQuantifier(quantifier=CvMy(), estimator_train=fh, estimator_test=fh)
        fh_cvmy.fit(X_train, y_train)

        #  PWK
        fh_knn_q = FrankAndHallQuantifier(quantifier=PWKQuantifier())
        fh_knn_q.fit(X_train, y_train)

        bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                                min_prevalence=None, random_state=current_seed)

        prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
        for n_bag in range(n_bags):

            prev_preds = [
                fh_cc.predict(X_test[indexes[:, n_bag], :]),
                fh_ac.predict(X_test[indexes[:, n_bag], :]),
                fh_pcc.predict(X_test[indexes[:, n_bag], :]),
                fh_pac.predict(X_test[indexes[:, n_bag], :]),
                fh_em.predict(X_test[indexes[:, n_bag], :]),
                fh_hdx.predict(X_test[indexes[:, n_bag], :]),
                fh_cdfx_l1.predict(X_test[indexes[:, n_bag], :]),
                fh_hdy.predict(X_test[indexes[:, n_bag], :]),
                fh_cdfy_l1.predict(X_test[indexes[:, n_bag], :]),
                fh_me.predict(X_test[indexes[:, n_bag], :]),
                fh_edx.predict(X_test[indexes[:, n_bag], :]),
                fh_edy.predict(X_test[indexes[:, n_bag], :]),
                fh_cvmy.predict(X_test[indexes[:, n_bag], :]),
                fh_knn_q.predict(X_test[indexes[:, n_bag], :])
            ]

            for n_method, prev_pred in enumerate(prev_preds):
                results[n_rep * n_bags + n_bag, n_method] = emd(prev_true[:, n_bag], prev_pred)

    #  printing and saving results
    filename = 'results/FrankAndHall-quantifiers'
    #  all
    np.savetxt(filename + '-all-' + str(n_reps) + '-' + str(n_bags), results,
               fmt='%.5f', delimiter=",", header=','.join(method_name))
    #  avg
    file_avg = open(filename + '-avg-' + str(n_reps) + '-' + str(n_bags), 'w')
    avg = np.mean(results, axis=0)
    print('\nEMD results')
    print('-' * 26)
    for n_method, method in enumerate(method_name):
        file_avg.write('%-19s%.5f\n' % (method, avg[n_method]))
        print('%-19s%.5f' % (method, avg[n_method]))


if __name__ == '__main__':
    main(dataset='datasets/ordinal/ESL.csv', n_reps=2, n_bags=50, master_seed=2032)

# EMD results
# --------------------------
# FH-CC              0.16721
# FH-AC              0.18131
# FH-PCC             0.18759
# FH-PAC             0.17923
# FH-EM              0.16481
# FH-HDX-8b          0.26857
# FH-CDFX-100b-L1    0.33637
# FH-HDy-8b          0.17344
# FH-CDFy-100b-L1    0.17652
# FH-FriedmanME-L1   0.21525
# FH-EDX             0.35678
# FH-EDy             0.18145
# FH-CvMy            0.25296
# FH-PWKQuantifier   0.21895
