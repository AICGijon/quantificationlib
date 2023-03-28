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

from data_utils import load_data, normalize



X, y = load_data("../datasets/multiclass/iris.csv")


# generating training-test partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2032)
X_train, X_test = normalize(X_train, X_test)
estimator = RandomForestClassifier(n_estimators=100, class_weight='balanced')
#  HDY
hdy = HDy(estimator_train=estimator, estimator_test=estimator, n_bins=8)
hdy.fit(X_train, y_train)

#  Testing bags
bag_generator = PriorShift_BagGenerator(n_bags=10, bag_size=len(X_test),
                                        min_prevalence=None, random_state=2032)

prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
for n_bag in range(10):
    print(hdy.predict(X_test[indexes[:, n_bag], :]))