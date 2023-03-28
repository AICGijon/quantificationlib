==========================================
Quantification Learning library for Python
==========================================

QuantificationLib is an open source library for **quantification learning**. Quantification is also known as prevalence estimation or class prior
estimation. It is a supervised machine learning task that involves training models to estimate the relative frequencies or prevalence values of 
the classes of interest in a samples (set of examples) of unlabelled data.

QuantificationLib implements a wide variety of binary and multiclass quantification methods. From well stablished baselines 
as *Classify and Count* or *Adjusted Count*, to more soffisticated methods like *distribution matching* methods or *ensembles*.

.. note::

   The library is designed with quick and efficient quantification experimentation in mind. The classifiers used by the quantifiers 
   are reused by the different quantifiers (no need to train them multiple times). Quantifiers `fit` methods have also the option to
   work directly with classifier predictions.


Quickstart
==========

:doc:`Installing`
In order to install the library, just use pip:

.. code-block:: bash

   pip install quantificationlib

.. note::

   We recomend to make the installation inside a virtual environment:

   .. code-block:: bash
      
      python3 -m venv venv
      source venv/bin/activate
      pip install quantificationlib


Now it is time to train your first quantifier:

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from quantificationlib.baselines.ac import AC
   from quantificationlib.bag_generator import PriorShift_BagGenerator
   from quantificationlib.metrics.multiclass import mean_absolute_error

   from data_utils import load_data,normalize

   X, y = load_data('../datasets/binary/iris.3.csv')

   # generating training-test partition
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
   X_train, X_test = normalize(X_train, X_test)

   #base classifier
   estimator = LogisticRegression()

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

This example is taken from https://github.com/AICGijon/quantificationlib/quantificationlib/examples. Check the link for more useful examples.

Library Summary
===============

Main Packages
-------------
.. toctree::
   :maxdepth: 1

   rsts_t4/quantificationlib.baselines
   rsts_t4/quantificationlib.binary
   rsts_t4/quantificationlib.decomposition
   rsts_t4/quantificationlib.ensembles
   rsts_t4/quantificationlib.estimators
   rsts_t4/quantificationlib.metrics
   rsts_t4/quantificationlib.multiclass
   rsts_t4/quantificationlib.ordinal

Other Modules
-------------

.. toctree::
   :maxdepth: 1

   rsts_t4/quantificationlib.bag_generator
   rsts_t4/quantificationlib.base
   rsts_t4/quantificationlib.cvxpy_installed_solvers
   rsts_t4/quantificationlib.optimization
   rsts_t4/quantificationlib.search


Main Packages Details
=====================
.. toctree::
   :maxdepth: 3

   rsts_t4/quantificationlib.baselines
   rsts_t4/quantificationlib.binary
   rsts_t4/quantificationlib.decomposition
   rsts_t4/quantificationlib.ensembles
   rsts_t4/quantificationlib.estimators
   rsts_t4/quantificationlib.metrics
   rsts_t4/quantificationlib.multiclass
   rsts_t4/quantificationlib.ordinal


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

