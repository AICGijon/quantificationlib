from sklearn.model_selection import (
    train_test_split,
)
from sklearn.linear_model import (
    LogisticRegression,
)
import urllib.request

from quantificationlib.baselines.ac import AC
from quantificationlib.multiclass.em import EM
from quantificationlib.bag_generator import (
    PriorShift_BagGenerator,
)
from quantificationlib.metrics.multiclass import (
    mean_absolute_error,
)
from quantificationlib.data_utils import (
    load_data,
    normalize,
)

# download data
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/AICGijon/quantificationlib/main/examples/datasets/binary/iris.3.csv",
    "datasets/binary/iris.3.csv",
)
X, y = load_data("datasets/binary/iris.3.csv")

# generating training-test partition
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=1,
)
X_train, X_test = normalize(X_train, X_test)

# base classifier
estimator = LogisticRegression()

# training Adjusted Count and EM quantifiers
ac = AC(
    estimator_train=estimator,
    estimator_test=estimator,
)
ac.fit(X_train, y_train)
em = EM(
    estimator_train=estimator,
    estimator_test=estimator,
)
em.fit(X_train, y_train)

# predicting a test sample
print(ac.predict(X=X_test))
print(em.predict(X=X_test))

bag_gen = PriorShift_BagGenerator(n_bags=10, bag_size=100)
prev_true, indexes = bag_gen.generate_bags(X_test, y_test)
preds = estimator.predict_proba(X_test)
for n_bag in range(10):
    p_hat_ac = ac.predict(
        X=None,
        predictions_test=preds[indexes[:, n_bag], :],
    )
    mae = mean_absolute_error(prev_true[:, n_bag], p_hat_ac)
    print("AC MAE=%f" % mae)
    p_hat_em = em.predict(
        X=None,
        predictions_test=preds[indexes[:, n_bag], :],
    )
    mae = mean_absolute_error(prev_true[:, n_bag], p_hat_em)
    print("EM MAE=%f" % mae)
