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

from quantificationlib.data_utils import load_data, normalize


def test_ordinal_quantifiers():

    method_name = ['PCC', 'OQT', 'AC_L2', 'ACOrd', 'PAC_L2', 'EDy_EMD', 'PDFOrd_L2', 'PDFOrd_EMD']

    dataset = 'examples/datasets/ordinal/ESL.csv'
    n_bags=5
    master_seed=2032

    results = np.zeros((n_bags, len(method_name)))

    X, y = load_data(dataset)

    current_seed = master_seed

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
            results[n_bag, n_method] = emd(prev_true[:, n_bag], prev_pred)

    avg = np.mean(results, axis=0)
    for name, result in zip(method_name,avg):
        assert result >= 0 and result <= 0.2, "Error in method %s " % name
