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

from quantificationlib.estimators.cross_validation import CV_estimator
from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.multiclass import hd, l1, l2, topsoe, mean_absolute_error

from quantificationlib.data_utils import load_data, normalize


def main(dataset, estimator_name, n_reps, n_bags, master_seed):

    method_name = ['CC', 'AC', 'PCC', 'PAC', 'EM',
                   'HDX-8b', 'CDFX-100b-L1', 'CDFX-100b-L2',
                   'HDy-8b', 'PDFy-8b-tipsoe',
                   'CDFy-100b-L1', 'CDFy-100b-L2', 'MMy-100', 'SORDy',
                   'DeBias', 'FriedmanME-L1',
                   'EDX', 'EDy', 'CvMy',
                   'QUANTy-L1', 'QUANTy-L2',
                   'REGX', 'REGy',
                   'PWKQuantifier']

    # method_name = ['REGX', 'Regy']

    results = np.zeros((n_reps * n_bags, len(method_name)))
    path, fname = os.path.split(dataset)
    print('Dataset: %s' % fname)
    X, y = load_data(dataset)

    for n_rep in range(n_reps):

        current_seed = master_seed + n_rep
        print("*** Training rep {}, seed {}".format(n_rep + 1, current_seed))

        # generating training-test partition
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                                                            random_state=current_seed)
        X_train, X_test = normalize(X_train, X_test)

        # classifiers are fitted by each object (all methods will use exactly the same predictions)
        # but they checked whether the estimator is already fitted (by a previous object) or not
        if estimator_name == 'RF':
            estimator = RandomForestClassifier(n_estimators=100, random_state=master_seed, class_weight='balanced')
        else:
            skf_train = StratifiedKFold(n_splits=10, shuffle=True, random_state=master_seed)
            estimator = CV_estimator(estimator=RandomForestClassifier(n_estimators=100, random_state=master_seed,
                                                                      class_weight='balanced'), cv=skf_train)

        #  AC
        ac = AC(estimator_train=estimator, estimator_test=estimator)
        ac.fit(X_train, y_train)
        #  CC
        cc = CC(estimator_test=estimator)
        cc.fit(X_train, y_train)
        #  PCC
        pcc = PCC(estimator_test=estimator)
        pcc.fit(X_train, y_train)
        #  PAC
        pac = PAC(estimator_train=estimator, estimator_test=estimator)
        pac.fit(X_train, y_train)

        # EM
        em = EM(estimator_train=estimator, estimator_test=estimator)
        em.fit(X_train, y_train)

        #  HDX
        hdx = HDX(n_bins=8)
        hdx.fit(X_train, y_train)

        #  CDFX-L1
        cdfx_l1 = DFX(distribution_function='CDF', n_bins=100, distance='L1')
        cdfx_l1.fit(X_train, y_train)
        #  CDFX-L2
        cdfx_l2 = DFX(distribution_function='CDF', n_bins=100, distance='L2')
        cdfx_l2.fit(X_train, y_train)

        #  HDY
        hdy = HDy(estimator_train=estimator, estimator_test=estimator, n_bins=8)
        hdy.fit(X_train, y_train)

        #  PDFY - topsoe
        pdfy_topsoe = DFy(estimator_train=estimator, estimator_test=estimator, distribution_function='PDF',
                          distance=topsoe, n_bins=8)
        pdfy_topsoe.fit(X_train, y_train)

        #  MMy
        mmy = MMy(estimator_train=estimator, estimator_test=estimator, n_bins=100)
        mmy.fit(X_train, y_train)

        #  SORDy
        sordy = SORDy(estimator_train=estimator, estimator_test=estimator)
        sordy.fit(X_train, y_train)

        #  CDFy-L1
        cdfy_l1 = DFy(estimator_train=estimator, estimator_test=estimator, distribution_function='CDF',
                      n_bins=100, distance='L1')
        cdfy_l1.fit(X_train, y_train)
        #  CDFy-L2
        cdfy_l2 = DFy(estimator_train=estimator, estimator_test=estimator, distribution_function='CDF',
                      n_bins=100, distance='L2')
        cdfy_l2.fit(X_train, y_train)

        #  Friedman DeBias
        db = DeBias(estimator_train=estimator, estimator_test=estimator)
        db.fit(X_train, y_train)
        #  FriedmanME - L1
        me_l1 = FriedmanME(estimator_train=estimator, estimator_test=estimator, distance='L1')
        me_l1.fit(X_train, y_train)

        #  EDX
        edx = EDX()
        edx.fit(X_train, y_train)
        #  EDy
        edy = EDy(estimator_train=estimator, estimator_test=estimator)
        edy.fit(X_train, y_train)
        #  CvMy
        cvmy = CvMy(estimator_train=estimator, estimator_test=estimator)
        cvmy.fit(X_train, y_train)

        #  QUANTy-L1
        quanty_l1 = QUANTy(estimator_train=estimator, estimator_test=estimator, n_quantiles=8, distance=l1)
        quanty_l1.fit(X_train, y_train)
        #  QUANTy-L2
        quanty_l2 = QUANTy(estimator_train=estimator, estimator_test=estimator, n_quantiles=8, distance=l2)
        quanty_l2.fit(X_train, y_train)

        #  REGX
        bag_generator = PriorShift_BagGenerator(n_bags=1000, bag_size=len(X_train),
                                                min_prevalence=None, random_state=master_seed)
        regx = REGX(bag_generator=bag_generator, n_bins=8, bin_strategy='equal_width', regression_estimator=None)
        regx.fit(X_train, y_train)
        # REGy
        regy = REGy(estimator_train=estimator, estimator_test=estimator, bag_generator=bag_generator,
                    n_bins=8, bin_strategy='equal_width', regression_estimator=None)
        regy.fit(X_train, y_train)

        # PWK
        knn_q = PWKQuantifier()
        knn_q.fit(X_train, y_train)

        #  Testing bags
        bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                                min_prevalence=None, random_state=current_seed)

        prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
        for n_bag in range(n_bags):

            prev_preds = [
                cc.predict(X_test[indexes[:, n_bag], :]),
                ac.predict(X_test[indexes[:, n_bag], :]),
                pcc.predict(X_test[indexes[:, n_bag], :]),
                pac.predict(X_test[indexes[:, n_bag], :]),
                em.predict(X_test[indexes[:, n_bag], :]),
                hdx.predict(X_test[indexes[:, n_bag], :]),
                cdfx_l1.predict(X_test[indexes[:, n_bag], :]),
                cdfx_l2.predict(X_test[indexes[:, n_bag], :]),
                hdy.predict(X_test[indexes[:, n_bag], :]),
                pdfy_topsoe.predict(X_test[indexes[:, n_bag], :]),
                mmy.predict(X_test[indexes[:, n_bag], :]),
                sordy.predict(X_test[indexes[:, n_bag], :]),
                cdfy_l1.predict(X_test[indexes[:, n_bag], :]),
                cdfy_l2.predict(X_test[indexes[:, n_bag], :]),
                db.predict(X_test[indexes[:, n_bag], :]),
                me_l1.predict(X_test[indexes[:, n_bag], :]),
                edx.predict(X_test[indexes[:, n_bag], :]),
                edy.predict(X_test[indexes[:, n_bag], :]),
                cvmy.predict(X_test[indexes[:, n_bag], :]),
                quanty_l1.predict(X_test[indexes[:, n_bag], :]),
                quanty_l2.predict(X_test[indexes[:, n_bag], :]),
                regx.predict(X_test[indexes[:, n_bag], :]),
                regy.predict(X_test[indexes[:, n_bag], :]),
                knn_q.predict(X_test[indexes[:, n_bag], :])
            ]

            for n_method, prev_pred in enumerate(prev_preds):
                results[n_rep * n_bags + n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

    #  printing and saving results
    filename = 'results/binary-quantifiers-' + estimator_name
    #  all
    np.savetxt(filename + '-all-' + str(n_reps) + '-' + str(n_bags), results,
               fmt='%.5f', delimiter=",", header='true_p' + ','.join(method_name))
    #  avg
    file_avg = open(filename + '-avg-' + str(n_reps) + '-' + str(n_bags), 'w')
    avg = np.mean(results, axis=0)
    print('\nMAE results')
    print('-' * 22)
    for n_method, method in enumerate(method_name):
        file_avg.write('%-15s%.5f\n' % (method, avg[n_method]))
        print('%-15s%.5f' % (method, avg[n_method]))


if __name__ == '__main__':
    main(dataset='datasets/binary/iris.3.csv', estimator_name='RF',
         n_reps=2, n_bags=50, master_seed=2032)

#  RF MAE results
# ----------------------
# CC             0.05089
# AC             0.05089
# PCC            0.04720
# PAC            0.03451
# EM             0.02301
# HDX-8b         0.03655
# CDFX-100b-L1   0.07567
# CDFX-100b-L2   0.06826
# HDy-8b         0.03218
# PDFy-8b-tipsoe 0.02675
# CDFy-100b-L1   0.03006
# CDFy-100b-L2   0.02984
# MMy-100        0.03006
# SORDy          0.03425
# DeBias         0.03310
# FriedmanME-L1  0.02833
# EDX            0.06821
# EDy            0.03417
# CvMy           0.06414
# QUANTy-L1      0.03229
# QUANTy-L2      0.03296
# REGX           0.04840
# REGy           0.03667
# PWKQuantifier  0.03578

#  CV MAE results
# ----------------------
# CC             0.05089
# AC             0.04208
# PCC            0.04898
# PAC            0.02844
# EM             0.02244
# HDX-8b         0.03655
# CDFX-100b-L1   0.07567
# CDFX-100b-L2   0.06826
# HDy-8b         0.05690
# PDFy-8b-tipsoe 0.04139
# CDFy-100b-L1   0.03039
# CDFy-100b-L2   0.03048
# MMy-100        0.03039
# SORDy          0.02805
# DeBias         0.03168
# FriedmanME-L1  0.03111
# EDX            0.06821
# EDy            0.02838
# CvMy           0.05104
# QUANTy-L1      0.02956
# QUANTy-L2      0.02820
# REGX           0.04840
# REGy           0.05790
# PWKQuantifier  0.03578
