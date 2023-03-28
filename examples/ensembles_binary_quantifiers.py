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

from quantificationlib.estimators.ensembles import EnsembleOfClassifiers
from quantificationlib.ensembles.eoq import EoQ

from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.multiclass import hd, l1, l2, topsoe, mean_absolute_error

from quantificationlib.examples.data_utils import load_data, normalize


def main(dataset, estimator_name, n_reps, n_bags, master_seed):

    method_name = ['EOQ-CC', 'EOQ-AC', 'EOQ-PCC', 'EOQ-PAC', 'EOQ-EM']
                   # 'EOQ-HDX-8b', 'EOQ-CDFX-100b-L1', 'EOQ-CDFX-100b-L2',
                   # 'EOQ-HDy-8b', 'EOQ-PDFy-8b-tipsoe',
                   # 'EOQ-CDFy-100b-L1', 'EOQ-CDFy-100b-L2', 'EOQ-SORDy',
                   # 'EOQ-DeBias', 'EOQ-FriedmanME-L1',
                   # 'EOQ-EDX', 'EOQ-EDy', 'EOQ-CvMy',
                   # 'EOQ-QUANTy-L1', 'EOQ-QUANTy-L2',
                   # 'EOQ-REGX', 'EOQ-REGy',
                   # 'EOQ-PWKQuantifier']

    results = np.zeros((n_reps * n_bags, len(method_name)))
    path, fname = os.path.split(dataset)
    print('Dataset: %s' % fname)
    X, y = load_data(dataset)

    for n_rep in range(n_reps):

        current_seed = master_seed + n_rep
        print("*** Training rep {}, seed {}".format(n_rep + 1, current_seed))

        # generating training-test partition
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                                                            random_state=current_seed)
        X_train, X_test = normalize(X_train, X_test)

        # classifiers are fitted by each object (all methods will use exactly the same predictions)
        # but they checked whether the estimator is already fitted (by a previous object) or not

        estimator = RandomForestClassifier(n_estimators=100, random_state=master_seed, class_weight='balanced')

        ensemble_estimator = EnsembleOfClassifiers(base_estimator=estimator)

        bag_generator_eoq = PriorShift_BagGenerator(n_bags=100, bag_size=len(X_train),
                                                    min_prevalence=0.05, random_state=current_seed)

        #  AC
        eoq_ac = EoQ(base_quantifier=AC(),
                     n_quantifiers=100, bag_generator=bag_generator_eoq,
                     combination_strategy='mean',
                     ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        eoq_ac.fit(X_train, y_train)
        #  CC
        eoq_cc = EoQ(base_quantifier=CC(),
                     n_quantifiers=100, bag_generator=bag_generator_eoq,
                     combination_strategy='mean',
                     ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        eoq_cc.fit(X_train, y_train)
        #  PCC
        eoq_pcc = EoQ(base_quantifier=PCC(),
                      n_quantifiers=100, bag_generator=bag_generator_eoq,
                      combination_strategy='mean',
                      ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        eoq_pcc.fit(X_train, y_train)
        #  PAC
        eoq_pac = EoQ(base_quantifier=PAC(),
                      n_quantifiers=100, bag_generator=bag_generator_eoq,
                      combination_strategy='mean',
                      ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        eoq_pac.fit(X_train, y_train)

        # EM
        eoq_em = EoQ(base_quantifier=EM(),
                     n_quantifiers=100, bag_generator=bag_generator_eoq,
                     combination_strategy='mean',
                     ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        eoq_em.fit(X_train, y_train)

        # #  HDX
        # eoq_hdx = EoQ(base_quantifier=HDX(n_bins=8),
        #               n_quantifiers=100, bag_generator=bag_generator_eoq,
        #               combination_strategy='mean',
        #               ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_hdx.fit(X_train, y_train)
        #
        # #  CDFX-L1
        # eoq_cdfx_l1 = EoQ(base_quantifier=DFX(distribution_function='CDF', n_bins=100, distance='L1'),
        #                   n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                   combination_strategy='mean',
        #                   ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_cdfx_l1.fit(X_train, y_train)
        # #  CDFX-L2
        # eoq_cdfx_l2 = EoQ(base_quantifier=DFX(distribution_function='CDF', n_bins=100, distance='L2'),
        #                   n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                   combination_strategy='mean',
        #                   ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_cdfx_l2.fit(X_train, y_train)
        #
        # #  HDY
        # eoq_hdy = EoQ(base_quantifier=HDy(n_bins=8),
        #               n_quantifiers=100, bag_generator=bag_generator_eoq,
        #               combination_strategy='mean',
        #               ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_hdy.fit(X_train, y_train)
        #
        # #  PDFY - topsoe
        # eoq_pdfy_topsoe = EoQ(base_quantifier=DFy(distribution_function='PDF', distance=topsoe, n_bins=8),
        #                       n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                       combination_strategy='mean',
        #                       ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_pdfy_topsoe.fit(X_train, y_train)
        #
        # #  SORDy
        # eoq_sordy = EoQ(base_quantifier=SORDy(),
        #                 n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                 combination_strategy='mean',
        #                 ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_sordy.fit(X_train, y_train)
        #
        # #  CDFy-L1
        # eoq_cdfy_l1 = EoQ(base_quantifier=DFy(distribution_function='CDF', n_bins=100, distance='L1'),
        #                   n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                   combination_strategy='mean',
        #                   ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_cdfy_l1.fit(X_train, y_train)
        # #  CDFy-L2
        # eoq_cdfy_l2 = EoQ(base_quantifier=DFy(distribution_function='CDF', n_bins=100, distance='L2'),
        #                   n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                   combination_strategy='mean',
        #                   ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_cdfy_l2.fit(X_train, y_train)
        #
        # #  Friedman DeBias
        # eoq_db = EoQ(base_quantifier=DeBias(),
        #              n_quantifiers=100, bag_generator=bag_generator_eoq,
        #              combination_strategy='mean',
        #              ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_db.fit(X_train, y_train)
        # #  FriedmanME - L1
        # eoq_me_l1 = EoQ(base_quantifier=FriedmanME(distance='L1'),
        #                 n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                 combination_strategy='mean',
        #                 ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_me_l1.fit(X_train, y_train)
        #
        # #  EDX
        # eoq_edx = EoQ(base_quantifier=EDX(),
        #               n_quantifiers=100, bag_generator=bag_generator_eoq,
        #               combination_strategy='mean',
        #               ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_edx.fit(X_train, y_train)
        # #  EDy
        # eoq_edy = EoQ(base_quantifier=EDy(),
        #               n_quantifiers=100, bag_generator=bag_generator_eoq,
        #               combination_strategy='mean',
        #               ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_edy.fit(X_train, y_train)
        # #  CvMy
        # eoq_cvmy = EoQ(base_quantifier=CvMy(),
        #                n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                combination_strategy='mean',
        #                ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_cvmy.fit(X_train, y_train)
        #
        # #  QUANTy-L1
        # eoq_quanty_l1 = EoQ(base_quantifier=QUANTy(n_quantiles=8, distance=l1),
        #                     n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                     combination_strategy='mean',
        #                     ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_quanty_l1.fit(X_train, y_train)
        # #  QUANTy-L2
        # eoq_quanty_l2 = EoQ(base_quantifier=QUANTy(n_quantiles=8, distance=l2),
        #                     n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                     combination_strategy='mean',
        #                     ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_quanty_l2.fit(X_train, y_train)
        #
        # #  REGX
        # bag_generator = PriorShift_BagGenerator(n_bags=1000, bag_size=len(X_train),
        #                                         min_prevalence=None, random_state=master_seed)
        # eoq_regx = EoQ(base_quantifier=REGX(bag_generator=bag_generator, n_bins=8,
        #                                     bin_strategy='equal_width', regression_estimator=None),
        #                n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                combination_strategy='mean',
        #                ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_regx.fit(X_train, y_train)
        # # REGy
        # eoq_regy = EoQ(base_quantifier=REGy(bag_generator=bag_generator, n_bins=8, bin_strategy='equal_width',
        #                                     regression_estimator=None),
        #                n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                combination_strategy='mean',
        #                ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_regy.fit(X_train, y_train)
        #
        # #  PWK
        # eoq_knn_q = EoQ(base_quantifier=PWKQuantifier(),
        #                 n_quantifiers=100, bag_generator=bag_generator_eoq,
        #                 combination_strategy='mean',
        #                 ensemble_estimator_train=ensemble_estimator, ensemble_estimator_test=ensemble_estimator)
        # eoq_knn_q.fit(X_train, y_train)

        #  Testing bags
        bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                                min_prevalence=None, random_state=current_seed)

        prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
        for n_bag in range(n_bags):

            prev_preds = [
                eoq_cc.predict(X_test[indexes[:, n_bag], :]),
                eoq_ac.predict(X_test[indexes[:, n_bag], :]),
                eoq_pcc.predict(X_test[indexes[:, n_bag], :]),
                eoq_pac.predict(X_test[indexes[:, n_bag], :]),
                eoq_em.predict(X_test[indexes[:, n_bag], :]),
                # eoq_hdx.predict(X_test[indexes[:, n_bag], :]),
                # eoq_cdfx_l1.predict(X_test[indexes[:, n_bag], :]),
                # eoq_cdfx_l2.predict(X_test[indexes[:, n_bag], :]),
                # eoq_hdy.predict(X_test[indexes[:, n_bag], :]),
                # eoq_pdfy_topsoe.predict(X_test[indexes[:, n_bag], :]),
                # eoq_sordy.predict(X_test[indexes[:, n_bag], :]),
                # eoq_cdfy_l1.predict(X_test[indexes[:, n_bag], :]),
                # eoq_cdfy_l2.predict(X_test[indexes[:, n_bag], :]),
                # eoq_db.predict(X_test[indexes[:, n_bag], :]),
                # eoq_me_l1.predict(X_test[indexes[:, n_bag], :]),
                # eoq_edx.predict(X_test[indexes[:, n_bag], :]),
                # eoq_edy.predict(X_test[indexes[:, n_bag], :]),
                # eoq_cvmy.predict(X_test[indexes[:, n_bag], :]),
                # eoq_quanty_l1.predict(X_test[indexes[:, n_bag], :]),
                # eoq_quanty_l2.predict(X_test[indexes[:, n_bag], :]),
                # eoq_regx.predict(X_test[indexes[:, n_bag], :]),
                # eoq_regy.predict(X_test[indexes[:, n_bag], :]),
                # eoq_knn_q.predict(X_test[indexes[:, n_bag], :])
            ]

            for n_method, prev_pred in enumerate(prev_preds):
                results[n_rep * n_bags + n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

    #  printing and saving results
    filename = 'results/ensembles-binary-quantifiers-' + estimator_name
    #  all
    np.savetxt(filename + '-all-' + str(n_reps) + '-' + str(n_bags), results,
               fmt='%.5f', delimiter=",", header='true_p' + ','.join(method_name))
    #  avg
    file_avg = open(filename + '-avg-' + str(n_reps) + '-' + str(n_bags), 'w')
    avg = np.mean(results, axis=0)
    print('\nMAE results')
    print('-' * 22)
    for n_method, method in enumerate(method_name):
        file_avg.write('%-15s%.5f\n' % (method, avg[n_method]))
        print('%-15s%.5f' % (method, avg[n_method]))


if __name__ == '__main__':
    main(dataset='../datasets/binary/iris.3.csv', estimator_name='RF', n_reps=2, n_bags=50, master_seed=2032)
