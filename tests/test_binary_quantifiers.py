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


def test_binary_quantifiers():

    dataset='examples/datasets/binary/iris.3.csv'
    n_bags=5
    master_seed=2032

    method_name = ['CC', 'AC', 'PCC', 'PAC', 'EM',
                   'HDX-8b', 'CDFX-100b-L1', 'CDFX-100b-L2',
                   'HDy-8b','HDy-8b-equal_with','HDy-8b-binormal','HDy-8b-normal', 
                   'PDFy-8b-tipsoe',
                   'CDFy-100b-L1', 'CDFy-100b-L2', 'MMy-100', 'SORDy',
                   'DeBias', 'FriedmanME-L1',
                   'EDX', 'EDy', 'CvMy',
                   'QUANTy-L1', 'QUANTy-L2',
                   'REGX', 'REGy',
                   'PWKQuantifier']

    results = np.zeros((n_bags, len(method_name)))
    X, y = load_data(dataset)

    current_seed = master_seed

    # generating training-test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=current_seed)
    X_train, X_test = normalize(X_train, X_test)

    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they checked whether the estimator is already fitted (by a previous object) or not
    estimator = RandomForestClassifier(n_estimators=100, random_state=master_seed, class_weight='balanced')

    #  AC
    ac = AC(estimator_train=estimator, estimator_test=estimator, verbose=1)
    ac.fit(X_train, y_train)
    #  CC
    cc = CC(estimator_test=estimator)
    cc.fit(X_train, y_train)
    #  PCC
    pcc = PCC(estimator_test=estimator)
    pcc.fit(X_train, y_train)
    #  PAC
    pac = PAC(estimator_train=estimator, estimator_test=estimator, verbose=1)
    pac.fit(X_train, y_train)

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

    #  HDY equal count
    hdy_equal_count = HDy(estimator_train=estimator, estimator_test=estimator, n_bins=8, bin_strategy='equal_count', verbose=1)
    hdy_equal_count.fit(X_train, y_train)

    #  HDY binormal
    hdy_binormal = HDy(estimator_train=estimator, estimator_test=estimator, n_bins=8, bin_strategy='binormal', verbose=1)
    hdy_binormal.fit(X_train, y_train)

    #  HDY normal
    hdy_normal = HDy(estimator_train=estimator, estimator_test=estimator, n_bins=8, bin_strategy='normal', verbose=1)
    hdy_normal.fit(X_train, y_train)

    #  PDFY - topsoe
    pdfy_topsoe = DFy(estimator_train=estimator, estimator_test=estimator, distribution_function='PDF',
                        distance=topsoe, n_bins=8, verbose=1)
    pdfy_topsoe.fit(X_train, y_train)

    #  MMy
    mmy = MMy(estimator_train=estimator, estimator_test=estimator, n_bins=100, verbose=1)
    mmy.fit(X_train, y_train)

    #  SORDy
    sordy = SORDy(estimator_train=estimator, estimator_test=estimator, verbose=1)
    sordy.fit(X_train, y_train)

    #  CDFy-L1
    cdfy_l1 = DFy(estimator_train=estimator, estimator_test=estimator, distribution_function='CDF',
                    n_bins=100, distance='L1', verbose=1)
    cdfy_l1.fit(X_train, y_train)
    #  CDFy-L2
    cdfy_l2 = DFy(estimator_train=estimator, estimator_test=estimator, distribution_function='CDF',
                    n_bins=100, distance='L2', verbose=1)
    cdfy_l2.fit(X_train, y_train)

    #  Friedman DeBias
    db = DeBias(estimator_train=estimator, estimator_test=estimator, verbose=1)
    db.fit(X_train, y_train)
    #  FriedmanME - L1
    me_l1 = FriedmanME(estimator_train=estimator, estimator_test=estimator, distance='L1', verbose=1)
    me_l1.fit(X_train, y_train)

    #  EDX
    edx = EDX(verbose=1)
    edx.fit(X_train, y_train)
    #  EDy
    edy = EDy(estimator_train=estimator, estimator_test=estimator, verbose=1)
    edy.fit(X_train, y_train)
    #  CvMy
    cvmy = CvMy(estimator_train=estimator, estimator_test=estimator, verbose=1)
    cvmy.fit(X_train, y_train)

    #  QUANTy-L1
    quanty_l1 = QUANTy(estimator_train=estimator, estimator_test=estimator, n_quantiles=8, distance=l1, verbose=1)
    quanty_l1.fit(X_train, y_train)
    #  QUANTy-L2
    quanty_l2 = QUANTy(estimator_train=estimator, estimator_test=estimator, n_quantiles=8, distance=l2, verbose=1)
    quanty_l2.fit(X_train, y_train)

    #  REGX
    bag_generator = PriorShift_BagGenerator(n_bags=1000, bag_size=len(X_train),
                                            min_prevalence=None, random_state=master_seed)
    regx = REGX(bag_generator=bag_generator, n_bins=8, bin_strategy='equal_width', regression_estimator=None, verbose=1)
    regx.fit(X_train, y_train)
    # REGy
    regy = REGy(estimator_train=estimator, estimator_test=estimator, bag_generator=bag_generator,
                n_bins=8, bin_strategy='equal_width', regression_estimator=None, verbose=1)
    regy.fit(X_train, y_train)

    # PWK
    knn_q = PWKQuantifier(verbose=1)
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
            hdy_equal_count.predict(X_test[indexes[:, n_bag], :]),
            hdy_binormal.predict(X_test[indexes[:, n_bag], :]),
            hdy_normal.predict(X_test[indexes[:, n_bag], :]),
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
            results[n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

    avg = np.mean(results, axis=0)
    for name, result in zip(method_name,avg):
        assert result >= 0 and result <= 0.08, "Error in method %s " % name




