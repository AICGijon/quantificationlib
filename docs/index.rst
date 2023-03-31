==========================================
Quantification Learning library for Python
==========================================

QuantificationLib is an open source library for **quantification learning**. Quantification is also known as prevalence estimation or class prior
estimation. It is a supervised machine learning task that involves training models to estimate the relative frequencies or prevalence values of 
the classes of interest in a samples (set of examples) of unlabelled data. Quantification is an important area of research with many applications
that vary from ecological analysis, social sciences, epidemiology, etc.

QuantificationLib implements a wide variety of binary and multiclass quantification methods. From well established baselines 
as *Classify and Count* or *Adjusted Count*, to more sophisticated methods like *distribution matching* methods or *ensembles*.

.. note::

   The library is designed with quick and efficient quantification experimentation in mind. The classifiers used by the quantifiers 
   are reused by the different quantifiers (no need to train them multiple times). Quantifiers `fit` methods have also the option to
   work directly with classifier predictions.

For an introduction to quantification, you can read the `quantification wikipedia page <https://en.wikipedia.org/wiki/Quantification_(machine_learning)>`_.
For a more in depth quantification text, we refer the practitioners to the paper:

González, P., Castaño, A., Chawla, N. V., & Coz, J. J. D. (2017). A review on quantification learning. ACM Computing Surveys (CSUR), 50(5), 1-40.  

Installation
============

In order to install the library you need Python 3. The library can be installed using pip:

.. code-block:: bash

   pip install quantificationlib

.. note::

   We recommend to make the installation inside a virtual environment:

   .. code-block:: bash
      
      python3 -m venv venv
      source venv/bin/activate
      pip install quantificationlib


Quickstart
==========

It is time to train your first quantifier:

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression

   from quantificationlib.baselines.ac import AC
   from quantificationlib.bag_generator import PriorShift_BagGenerator
   from quantificationlib.metrics.multiclass import mean_absolute_error

   from quantificationlib.data_utils import load_data,normalize

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

The previous example will train an *Adjusted Count* quantifier over the 
data and quantify 10 test bags, printing the real and estimated prevalence 
of each one, along with the absolute error of the quantifier in each bag:

.. code-block::

   True prevalence=0.688889, estimated prevalence=0.619048, AE=0.069841
   True prevalence=0.355556, estimated prevalence=0.298060, AE=0.057496
   True prevalence=0.400000, estimated prevalence=0.347443, AE=0.052557
   True prevalence=0.177778, estimated prevalence=0.149912, AE=0.027866
   True prevalence=0.555556, estimated prevalence=0.396825, AE=0.158730
   True prevalence=0.222222, estimated prevalence=0.174603, AE=0.047619
   True prevalence=0.755556, estimated prevalence=0.668430, AE=0.087125
   True prevalence=0.111111, estimated prevalence=0.051146, AE=0.059965
   True prevalence=0.111111, estimated prevalence=0.075838, AE=0.035273
   True prevalence=0.844444, estimated prevalence=0.767196, AE=0.077249


This example is taken from `here <https://github.com/AICGijon/quantificationlib/examples>`_. Check the link for more useful examples.

Library Summary
===============

Packages
--------
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

Modules
-------

.. toctree::
   :maxdepth: 1

   rsts_t4/quantificationlib.base
   rsts_t4/quantificationlib.bag_generator
   rsts_t4/quantificationlib.cvxpy_installed_solvers
   rsts_t4/quantificationlib.optimization
   rsts_t4/quantificationlib.search


Packages Details
================
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

