import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.baselines.cc import CC, PCC

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

def test_multiclass_quantifiers():
    dataset = 'examples/datasets/multiclass/iris.csv'
    n_bags=5
    master_seed=2032
    method_name = ['CC', 'AC-HD', 'AC-L1', 'AC-L2', 'PCC', 'PAC-HD', 'PAC-L1', 'PAC-L2', 'EM',
                   'HDX-8b', 'CDFX-100b-L1', 'CDFX-100b-L2',
                   'HDy-8b', 'MMy-100', 'CDFy-100b-L1', 'CDFy-100b-L2',
                   'FriedmanME-HD', 'FriedmanME-L1', 'FriedmanME-L2',
                   'EDX', 'EDy', 'CvMy', 'REGX', 'REGy',
                   'PWKQuantifier']

    results = np.zeros((n_bags, len(method_name)))
    X, y = load_data(dataset)

    current_seed = master_seed

    # generating training-test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=current_seed)
    X_train, X_test = normalize(X_train, X_test)

    estimator = RandomForestClassifier(n_estimators=100, random_state=master_seed, class_weight='balanced')

    #  AC_HD
    ac_hd = AC(estimator_train=estimator, estimator_test=estimator, distance='HD', verbose=1)
    ac_hd.fit(X_train, y_train)
    #  AC_L1
    ac_l1 = AC(estimator_train=estimator, estimator_test=estimator, distance='L1', verbose=1)
    ac_l1.fit(X_train, y_train)
    #  AC_L2
    ac_l2 = AC(estimator_train=estimator, estimator_test=estimator, distance='L2', verbose=1)
    ac_l2.fit(X_train, y_train)
    #  CC
    cc = CC(estimator_test=estimator)
    cc.fit(X_train, y_train)
    #  PCC
    pcc = PCC(estimator_test=estimator)
    pcc.fit(X_train, y_train)
    #  PAC_HD
    pac_hd = PAC(estimator_train=estimator, estimator_test=estimator, distance='HD', verbose=1)
    pac_hd.fit(X_train, y_train)
    #  PAC_L1
    pac_l1 = PAC(estimator_train=estimator, estimator_test=estimator, distance='L1', verbose=1)
    pac_l1.fit(X_train, y_train)
    #  PAC_L2
    pac_l2 = PAC(estimator_train=estimator, estimator_test=estimator, distance='L2', verbose=1)
    pac_l2.fit(X_train, y_train)

    # EM
    em = EM(estimator_train=estimator, estimator_test=estimator, verbose=1)
    em.fit(X_train, y_train)

    #  HDX
    hdx = HDX(n_bins=8)
    hdx.fit(X_train, y_train)

    #  CDFX-L1
    cdfx_l1 = DFX(distribution_function='CDF', n_bins=100, distance='L1', verbose=1)
    cdfx_l1.fit(X_train, y_train)
    #  CDFX-L2
    cdfx_l2 = DFX(distribution_function='CDF', n_bins=100, distance='L2', verbose=1)
    cdfx_l2.fit(X_train, y_train)

    #  HDY
    hdy = HDy(estimator_train=estimator, estimator_test=estimator, n_bins=8, verbose=1)
    hdy.fit(X_train, y_train)

    #  MMy
    mmy = MMy(estimator_train=estimator, estimator_test=estimator, n_bins=100, verbose=1)
    mmy.fit(X_train, y_train)

    #  CDFy-L1
    cdfy_l1 = DFy(estimator_train=estimator, estimator_test=estimator, distribution_function='CDF',
                    n_bins=100, distance='L1', verbose=1)
    cdfy_l1.fit(X_train, y_train)
    #  CDFy-L2
    cdfy_l2 = DFy(estimator_train=estimator, estimator_test=estimator, distribution_function='CDF',
                    n_bins=100, distance='L2', verbose=1)
    cdfy_l2.fit(X_train, y_train)

    #  FriedmanME - HD
    me_hd = FriedmanME(estimator_train=estimator, estimator_test=estimator, distance='HD', verbose=1)
    me_hd.fit(X_train, y_train)
    #  FriedmanME - L1
    me_l1 = FriedmanME(estimator_train=estimator, estimator_test=estimator, distance='L1', verbose=1)
    me_l1.fit(X_train, y_train)
    #  FriedmanME - L2
    me_l2 = FriedmanME(estimator_train=estimator, estimator_test=estimator, distance='L2', verbose=1)
    me_l2.fit(X_train, y_train)

    #  EDX
    edx = EDX(verbose=1)
    edx.fit(X_train, y_train)
    #  EDy
    edy = EDy(estimator_train=estimator, estimator_test=estimator, verbose=1)
    edy.fit(X_train, y_train)
    #  CvMy
    cvmy = CvMy(estimator_train=estimator, estimator_test=estimator, verbose=1)
    cvmy.fit(X_train, y_train)

    #  REGX
    bag_generator = PriorShift_BagGenerator(n_bags=1000, bag_size=len(X_train),
                                            min_prevalence=None, random_state=master_seed)
    regx = REGX(bag_generator=bag_generator, n_bins=8, bin_strategy='equal_width', regression_estimator=None, verbose=1)
    regx.fit(X_train, y_train)
    # REGy
    regy = REGy(estimator_train=estimator, estimator_test=estimator, bag_generator=bag_generator,
                n_bins=8, bin_strategy='equal_width', regression_estimator=None, verbose=1)
    regy.fit(X_train, y_train)

    #  PWK
    knn_q = PWKQuantifier(verbose=1)
    knn_q.fit(X_train, y_train)

    #  Testing bags
    bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                            min_prevalence=None, random_state=current_seed)

    prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
    for n_bag in range(n_bags):

        prev_preds = [
            cc.predict(X_test[indexes[:, n_bag], :]),
            ac_hd.predict(X_test[indexes[:, n_bag], :]),
            ac_l1.predict(X_test[indexes[:, n_bag], :]),
            ac_l2.predict(X_test[indexes[:, n_bag], :]),
            pcc.predict(X_test[indexes[:, n_bag], :]),
            pac_hd.predict(X_test[indexes[:, n_bag], :]),
            pac_l1.predict(X_test[indexes[:, n_bag], :]),
            pac_l2.predict(X_test[indexes[:, n_bag], :]),
            em.predict(X_test[indexes[:, n_bag], :]),
            hdx.predict(X_test[indexes[:, n_bag], :]),
            cdfx_l1.predict(X_test[indexes[:, n_bag], :]),
            cdfx_l2.predict(X_test[indexes[:, n_bag], :]),
            hdy.predict(X_test[indexes[:, n_bag], :]),
            mmy.predict(X_test[indexes[:, n_bag], :]),
            cdfy_l1.predict(X_test[indexes[:, n_bag], :]),
            cdfy_l2.predict(X_test[indexes[:, n_bag], :]),
            me_hd.predict(X_test[indexes[:, n_bag], :]),
            me_l1.predict(X_test[indexes[:, n_bag], :]),
            me_l2.predict(X_test[indexes[:, n_bag], :]),
            edx.predict(X_test[indexes[:, n_bag], :]),
            edy.predict(X_test[indexes[:, n_bag], :]),
            cvmy.predict(X_test[indexes[:, n_bag], :]),
            regx.predict(X_test[indexes[:, n_bag], :]),
            regy.predict(X_test[indexes[:, n_bag], :]),
            knn_q.predict(X_test[indexes[:, n_bag], :])
        ]

        for n_method, prev_pred in enumerate(prev_preds):
            results[n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

   
    avg = np.mean(results, axis=0)
    for name, result in zip(method_name,avg):
        assert result >= 0 and result <= 0.1, "Error in method %s " % name