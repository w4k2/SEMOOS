import numpy as np
import strlearn as sl
import warnings
import os
import time
from joblib import Parallel, delayed
import logging
import traceback

from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from utils.load_dataset import load_data, find_datasets
from methods.moo_ensemble_bootstrap import MooEnsembleSVCbootstrap


logging.basicConfig(filename='textinfo/experiment0_set_featboot.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
logging.info("--------------------------------------------------------------------------------")
logging.info("-------                        NEW EXPERIMENT                            -------")
logging.info("--------------------------------------------------------------------------------")


def compute(dataset_id, dataset, methods, n_folds, metrics, metrics_alias, n_rows_p):
    logging.basicConfig(filename='textinfo/experiment0_set_featboot.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
    try:
        warnings.filterwarnings("ignore")
        print("START: %s" % (dataset))
        logging.info("START - %s" % (dataset))
        start = time.time()

        dataset_path = "datasets/set_params/" + dataset + ".dat"
        X, y, classes = load_data(dataset_path)
        X = X.to_numpy()
        # Normalization - transform data to [0, 1]
        X = MinMaxScaler().fit_transform(X, y)

        scores = np.zeros((len(metrics), len(methods), n_folds))
        diversity = np.zeros((len(methods), n_folds, 4))
        pareto_solutions = np.zeros((n_folds, n_rows_p, 2))

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            IR = {}
            for class_num in set(y_train):
                IR[class_num] = np.sum(y_train == class_num)

            for clf_id, clf_name in enumerate(methods):
                clf = clone(methods[clf_name])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                for metric_id, metric in enumerate(metrics):
                    scores[metric_id, clf_id, fold_id] = metric(y_test, y_pred)
                calculate_diversity = getattr(clf, "calculate_diversity", None)
                if callable(calculate_diversity):
                    diversity[clf_id, fold_id] = clf.calculate_diversity()
                else:
                    diversity[clf_id, fold_id] = None

                if hasattr(clf, 'solutions'):
                    for sol_id, solution in enumerate(clf.solutions):
                        for s_id, s in enumerate(solution):
                            pareto_solutions[fold_id, sol_id, s_id] = s
        # Save results to csv
        for clf_id, clf_name in enumerate(methods):
            for metric_id, metric in enumerate(metrics_alias):
                # Save metric results
                filename = "results/experiment0_set_featboot/raw_results/%s/%s/%s.csv" % (metric, dataset, clf_name)
                if not os.path.exists("results/experiment0_set_featboot/raw_results/%s/%s/" % (metric, dataset)):
                    os.makedirs("results/experiment0_set_featboot/raw_results/%s/%s/" % (metric, dataset))
                np.savetxt(fname=filename, fmt="%f", X=scores[metric_id, clf_id, :])
            # Save diversity results
            filename = "results/experiment0_set_featboot/diversity_results/%s/%s.csv" % (dataset, clf_name)
            if not os.path.exists("results/experiment0_set_featboot/diversity_results/%s/" % (dataset)):
                os.makedirs("results/experiment0_set_featboot/diversity_results/%s/" % (dataset))
            np.savetxt(fname=filename, fmt="%f", X=diversity[clf_id, :, :])
        # Save results pareto_solutions to csv
        for fold_id in range(n_folds):
            for sol_id in range(n_rows_p):
                if (pareto_solutions[fold_id, sol_id, 0] != 0.0) and (pareto_solutions[fold_id, sol_id, 1] != 0.0):
                    filename_pareto = "results/experiment0_set_featboot/pareto_raw/%s/fold%d/sol%d.csv" % (dataset, fold_id, sol_id)
                    if not os.path.exists("results/experiment0_set_featboot/pareto_raw/%s/fold%d/" % (dataset, fold_id)):
                        os.makedirs("results/experiment0_set_featboot/pareto_raw/%s/fold%d/" % (dataset, fold_id))
                    np.savetxt(fname=filename_pareto, fmt="%f", X=pareto_solutions[fold_id, sol_id, :])

        end = time.time() - start
        logging.info("DONE - %s (Time: %d [s])" % (dataset, end))
        print("DONE - %s (Time: %d [s])" % (dataset, end))

    except Exception as ex:
        logging.exception("Exception in %s" % (dataset))
        print("ERROR: %s" % (dataset))
        traceback.print_exc()
        print(str(ex))


base_estimator = SVC(probability=True)
methods = {
    "MooEnsembleSVCbootstrap_b1_f25": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5, n_repeats=1, scale_features=0.25),
    "MooEnsembleSVCbootstrap_b1_f50": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5, n_repeats=1, scale_features=0.5),
    "MooEnsembleSVCbootstrap_b1_f75": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5, n_repeats=1, scale_features=0.75),
    "MooEnsembleSVCbootstrap_b5_f25": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5, n_repeats=5, scale_features=0.25),
    "MooEnsembleSVCbootstrap_b5_f50": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5, n_repeats=5, scale_features=0.5),
    "MooEnsembleSVCbootstrap_b5_f75": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5, n_repeats=5, scale_features=0.75),
    "MooEnsembleSVCbootstrap_b10_f25": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5, n_repeats=10, scale_features=0.25),
    "MooEnsembleSVCbootstrap_b10_f50": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5, n_repeats=10, scale_features=0.5),
    "MooEnsembleSVCbootstrap_b10_f75": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5, n_repeats=10, scale_features=0.75),
    }

# Repeated Stratified K-Fold cross validator
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
n_folds = n_splits * n_repeats

"""
Imbalanced datasets from KEEL:
    - Imbalance ratio higher than 9 - part 1
    - IR (imbalance ratio) = majority / minority
        higher IR, higher imbalance of the dataset
"""
DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/set_params')

metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.geometric_mean_score_1, sl.metrics.geometric_mean_score_2, sl.metrics.f1_score, sl.metrics.recall, sl.metrics.specificity, sl.metrics.precision]
metrics_alias = ["BAC", "Gmean", "Gmean2", "F1score", "Recall", "Specificity", "Precision"]

n_rows_p = 100

# Multithread; n_jobs - number of threads, where -1 all threads, safe for my computer 2
Parallel(n_jobs=-1)(
                delayed(compute)
                (dataset_id, dataset, methods, n_folds, metrics, metrics_alias, n_rows_p)
                for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR))
                )

logging.info("-------------------")
logging.info("EXPERIMENT FINISHED")
logging.info("-------------------")
