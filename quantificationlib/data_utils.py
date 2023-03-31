"""
Basic tools for data loading and normalization. Basically used in the examples and tests.
"""

import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning

warnings.simplefilter("ignore", DataConversionWarning)


def normalize(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def load_data(dfile):
    data = np.genfromtxt(dfile, delimiter=',')

    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y