from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import urllib.request

from quantificationlib.baselines.ac import AC
from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.multiclass import mean_absolute_error
from quantificationlib.data_utils import load_data,normalize

#download data
urllib.request.urlretrieve("https://raw.githubusercontent.com/AICGijon/quantificationlib/main/examples/datasets/binary/iris.3.csv", "iris.3.csv")
X, y = load_data('datasets/binary/iris.3.csv')

# generating training-test partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
X_train, X_test = normalize(X_train, X_test)

#base classifier
estimator = LogisticRegression(random_state=1)

#quantifier
ac = AC(estimator_train=estimator, estimator_test=estimator)
ac.fit(X_train, y_train)
    
#  Testing bags
bag_generator = PriorShift_BagGenerator(n_bags=10, bag_size=len(X_test),min_prevalence=None)

prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
for n_bag in range(10):
    prev_pred = ac.predict(X_test[indexes[:, n_bag], :])
    print('True prevalence=%f, estimated prevalence=%f, AE=%f' % 
          (prev_true[1,n_bag],prev_pred[1], 
           mean_absolute_error(prev_true[:,n_bag],prev_pred)))