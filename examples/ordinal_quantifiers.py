import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestClassifier

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.baselines.cc import PCC

from quantificationlib.multiclass.energy import EDy

from quantificationlib.ordinal.pdf import PDFOrdinaly
from quantificationlib.ordinal.ac_ordinal import ACOrdinal
from quantificationlib.ordinal.trees import OrdinalQuantificationTree

from quantificationlib.estimators.frank_and_hall import FrankAndHallTreeClassifier

from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.ordinal import emd, emd_distances

from quantificationlib.examples.data_utils import load_data, normalize


def main(dataset, n_reps, n_bags, master_seed):

    method_name = ['PCC', 'OQT', 'AC_L2', 'ACOrd', 'PAC_L2', 'EDy_EMD', 'PDFOrd_L2', 'PDFOrd_EMD']

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
        fh = FrankAndHallTreeClassifier(estimator=RandomForestClassifier(n_estimators=100,
                                                                         random_state=master_seed,
                                                                         class_weight='balanced'), n_jobs=-1)

        #  PCC
        pcc = PCC(estimator_test=fh)
        pcc.fit(X_train, y_train)

        #  OQT
        oqt = OrdinalQuantificationTree(estimator_test=RandomForestClassifier(n_estimators=100,
                                                                              random_state=master_seed,
                                                                              class_weight='balanced'))
        oqt.fit(X_train, y_train)

        #  AC L2
        ac_l2 = AC(estimator_train=fh, estimator_test=fh, distance='L2')
        ac_l2.fit(X_train, y_train)

        #  AC ORD
        ac_ord = ACOrdinal(estimator_train=fh, estimator_test=fh)
        ac_ord.fit(X_train, y_train)

        #  PAC L2
        pac_l2 = PAC(estimator_train=fh, estimator_test=fh, distance='L2')
        pac_l2.fit(X_train, y_train)

        #  EDy
        edy = EDy(estimator_train=fh, estimator_test=fh, distance=emd_distances)
        edy.fit(X_train, y_train)

        #  PDF L2
        pdf_l2 = PDFOrdinaly(estimator_train=fh, estimator_test=fh, distance='L2')
        pdf_l2.fit(X_train, y_train)

        #  PDF EMD
        pdf_emd = PDFOrdinaly(estimator_train=fh, estimator_test=fh, distance='EMD')
        pdf_emd.fit(X_train, y_train)

        bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                                min_prevalence=None, random_state=current_seed)

        prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
        for n_bag in range(n_bags):

            prev_preds = [
                pcc.predict(X_test[indexes[:, n_bag], :]),
                oqt.predict(X_test[indexes[:, n_bag], :]),
                ac_l2.predict(X_test[indexes[:, n_bag], :]),
                ac_ord.predict(X_test[indexes[:, n_bag], :]),
                pac_l2.predict(X_test[indexes[:, n_bag], :]),
                edy.predict(X_test[indexes[:, n_bag], :]),
                pdf_l2.predict(X_test[indexes[:, n_bag], :]),
                pdf_emd.predict(X_test[indexes[:, n_bag], :])
            ]

            for n_method, prev_pred in enumerate(prev_preds):
                results[n_rep * n_bags + n_bag, n_method] = emd(prev_true[:, n_bag], prev_pred)

    #  printing and saving results
    filename = 'results/ordinal-quantifiers'
    #  all
    np.savetxt(filename + '-all-' + str(n_reps) + '-' + str(n_bags), results,
               fmt='%.5f', delimiter=",", header=','.join(method_name))
    #  avg
    file_avg = open(filename + '-avg-' + str(n_reps) + '-' + str(n_bags), 'w')
    avg = np.mean(results, axis=0)
    print('\nEMD results')
    print('-' * 22)
    for n_method, method in enumerate(method_name):
        file_avg.write('%-15s%.5f\n' % (method, avg[n_method]))
        print('%-15s%.5f' % (method, avg[n_method]))


if __name__ == '__main__':
    main(dataset='../datasets/ordinal/ESL.csv', n_reps=2, n_bags=50, master_seed=2032)


# EMD results
# ----------------------
# PCC            0.18793
# OQT            0.18793
# AC_L2          0.16564
# ACOrd          0.16557
# PAC_L2         0.15048
# EDy_EMD        0.15108
# PDFOrd_L2      0.19433
# PDFOrd_EMD     0.15610
