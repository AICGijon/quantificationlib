"""
Ordinal quantification trees
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

from quantificationlib.baselines.cc import PCC
from quantificationlib.estimators.frank_and_hall import FrankAndHallTreeClassifier


class OrdinalQuantificationTree(PCC):
    """ Ordinal Quantification Trees proposed by (Martino, Gao and Sebastiani, 2016)

        This class is just a wrapper. It is a PCC method in which the estimator for the test distribution is a
        FrankAndHallTreeClassifier. Notice that the estimator object for the OrdinalQuantificationTree must be
        the binary base estimator used by the FrankAndHallTreeClassifier

        Instead of using this class, our recommedation is to employ directly a PCC object to improve the efficiency
        because the FrankAndHallTreeClassifier estimator can be shared with other quantifiers and trained just once.

        References
        ----------
        Giovanni Da San Martino, Wei Gao, and Fabrizio Sebastiani. 2016a. Ordinal text quantification.
        In Proceedings of the International ACM SIGIR Conference on  Research and Development
        in Information Retrieval. 937940.

        Giovanni Da San Martino,Wei Gao, and Fabrizio Sebastiani. 2016b.
        QCRI at SemEval-2016 Task 4: Probabilistic methods for binary and ordinal quantification.
        In Proceedings of the 10th InternationalWorkshop on Semantic Evaluation (SemEval’16).
        Association for Computational Linguistics, A, 5863.
    """
    def __init__(self, estimator_test=None, verbose=0):
        super(OrdinalQuantificationTree, self).__init__(estimator_test=FrankAndHallTreeClassifier(estimator_test),
                                                        verbose=verbose)
