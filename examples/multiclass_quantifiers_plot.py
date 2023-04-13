import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.baselines.cc import CC, PCC
from quantificationlib.multiclass.energy import EDX, EDy, CvMy
from quantificationlib.multiclass.df import HDX, HDy, MMy, DFX, DFy
from quantificationlib.multiclass.em import EM

from quantificationlib.estimators.cross_validation import CV_estimator
from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.metrics.multiclass import hd, l1, l2, topsoe, mean_absolute_error

from quantificationlib.data_utils import load_data, normalize

from quantificationlib.plot.qlplot import plot_line_prevalences     ### To plot prevalences
import matplotlib.pyplot as plt                                     ### To plot prevalences

def main(dataset, estimator_name, n_reps, n_bags, master_seed):

    method_name = ['PCC', 'HDY', 'EM']

    results = np.zeros((n_reps * n_bags, len(method_name)))

    path, fname = os.path.split(dataset)
    print('Dataset: %s' % fname)
    X, y = load_data(dataset)

    ### To plot prevalences
    n_classes = len(np.unique(y)) 
    total_prev_true = np.zeros((0, n_classes))  
    total_prev_preds = np.zeros((0, n_classes)) 

    for n_rep in range(n_reps):

        current_seed = master_seed + n_rep
        print("*** Training rep {}, seed {}".format(n_rep + 1, current_seed))

        # generating training-test partition
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=current_seed)
        X_train, X_test = normalize(X_train, X_test)

        # classifiers are fitted by each object (all methods will use exactly the same predictions)
        # but they checked whether the estimator is already fitted (by a previous object) or not
        if estimator_name == 'RF':
            estimator = RandomForestClassifier(n_estimators=100, random_state=master_seed, class_weight='balanced')
        else:
            skf_train = StratifiedKFold(n_splits=10, shuffle=True, random_state=master_seed)
            estimator = CV_estimator(estimator=RandomForestClassifier(n_estimators=100, random_state=master_seed,
                                                                      class_weight='balanced'), cv=skf_train)


         #  PCC
        pcc = PCC(estimator_test=estimator)
        pcc.fit(X_train, y_train)
        #  HDY
        hdy = HDy(estimator_train=estimator, estimator_test=estimator, n_bins=8)
        hdy.fit(X_train, y_train)
        #  EM
        em = EM(estimator_train=estimator, estimator_test=estimator)
        em.fit(X_train, y_train)

        #  Testing bags
        bag_generator = PriorShift_BagGenerator(n_bags=n_bags, bag_size=len(X_test),
                                                min_prevalence=None, random_state=current_seed)

        prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
        
        ### To plot prevalences
        total_prev_true = np.concatenate((total_prev_true, prev_true.T), axis=0)  

        for n_bag in range(n_bags):

            prev_preds = [
                pcc.predict(X_test[indexes[:, n_bag], :]),
                hdy.predict(X_test[indexes[:, n_bag], :]),
                em.predict(X_test[indexes[:, n_bag], :])
            ]
   
            ### To plot prevalences
            total_prev_preds = np.concatenate((total_prev_preds, prev_preds), axis=0) 

            for n_method, prev_pred in enumerate(prev_preds):
                results[n_rep * n_bags + n_bag, n_method] = mean_absolute_error(prev_true[:, n_bag], prev_pred)

    #  printing and saving results
    filename = 'results/multiclass-quantifiers-' + estimator_name
    #  all
    np.savetxt(filename + '-all-' + str(n_reps) + '-' + str(n_bags), results,
               fmt='%.5f', delimiter=",", header=','.join(method_name))
    #  avg
    file_avg = open(filename + '-avg-' + str(n_reps) + '-' + str(n_bags), 'w')
    avg = np.mean(results, axis=0)
    print('\nMAE results')
    print('-' * 22)
    for n_method, method in enumerate(method_name):
        file_avg.write('%-15s%.5f\n' % (method, avg[n_method]))
        print('%-15s%.5f' % (method, avg[n_method]))

    # Create a single row of subplots
    fig, axs = plt.subplots(ncols=len(method_name)+1, figsize=(15, 4))

    # Plot the truth data in the first subplot
    plot_line_prevalences(axs[0], total_prev_true, order="descending")
    axs[0].set_title("TRUTH")

    # Loop through each method and plot the data in the corresponding subplot
    for i, name in enumerate(method_name):
        plot_line_prevalences(axs[i+1], total_prev_preds[i::len(method_name), :], order="descending")
        axs[i+1].set_title("%s (AE=%.5f)" % (name,avg[i]))

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
     #main(dataset='./datasets/multiclass/iris.csv', estimator_name='RF',
     #     n_reps=2, n_bags=100, master_seed=2032)

     main(dataset='./datasets/multiclass/MaternalHealthRisk.csv', estimator_name='RF',
          n_reps=4, n_bags=50, master_seed=2032)
