import os
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from methods.moo_ensemble import MooEnsembleSVC
from methods.moo_ensemble_bootstrap import MooEnsembleSVCbootstrap
from methods.moo_ensemble_bootstrap_pruned import MooEnsembleSVCbootstrapPruned
from methods.random_subspace_ensemble import RandomSubspaceEnsemble
from methods.feature_selection_clf import FeatueSelectionClf
from utils.load_dataset import find_datasets, calc_imbalance_ratio


warnings.filterwarnings("ignore")


base_estimator = {'SVM': SVC(probability=True)}
# IR is an example, not real values of datasets
IR_class = {0: 1, 1: 1}

methods = {
    "MooEnsembleSVC": MooEnsembleSVC(base_classifier=base_estimator),
    "MooEnsembleSVCbootstrap": MooEnsembleSVCbootstrap(base_classifier=base_estimator),
    "MooEnsembleSVCbootstrapPruned": MooEnsembleSVCbootstrapPruned(base_classifier=base_estimator),
    "RandomSubspace": RandomSubspaceEnsemble(base_classifier=base_estimator),
    "SVM": SVC(),
    "FS": FeatueSelectionClf(base_estimator, chi2),
    "FSIRSVM": FeatueSelectionClf(SVC(kernel='linear', class_weight=IR_class), chi2)

}

methods_alias = [
                "SEMOOS",
                "SEMOOSb",
                "SEMOOSbp",
                "RS",
                "SVM",
                "FS",
                "FSIRSVM"
                ]

metrics_alias = ["BAC", "Gmean", "Gmean2", "F1score", "Recall", "Specificity", "Precision"]

n_splits = 2
n_repeats = 5
n_folds = n_splits * n_repeats
n_methods = len(methods_alias) * len(base_estimator)
n_metrics = len(metrics_alias)

directories = ["9lower", "9higher_part1", "9higher_part2", "9higher_part3"]

n_datasets = 0
datasets = []
for dir_id, dir in enumerate(directories):
    DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s/' % dir)
    n_datasets += len(list(enumerate(find_datasets(DATASETS_DIR))))
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        datasets.append(dataset)

mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))
experiments_paths = ["experiment4_9lower", "experiment1_9higher_part1", "experiment2_9higher_part2", "experiment3_9higher_part3"]
for exp in experiments_paths:
    for dataset_id, dataset in enumerate(datasets):
        for clf_id, clf_name in enumerate(methods):
            for metric_id, metric in enumerate(metrics_alias):
                try:
                    filename = "results/experiment_server/%s/raw_results/%s/%s/%s.csv" % (exp, metric, dataset, clf_name)
                    if not os.path.isfile(filename):
                        # print("File not exist - %s" % filename)
                        continue
                    scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    mean_score = np.mean(scores)
                    mean_scores[dataset_id, metric_id, clf_id] = mean_score
                    std = np.std(scores)
                    stds[dataset_id, metric_id, clf_id] = std
                except:
                    print("Error loading dataset - %s!" % dataset)

IR = calc_imbalance_ratio(directories=directories)
IR_argsorted = np.argsort(IR)

# Save dataset name, mean scores and standard deviation to .tex file
for metric_id, metric in enumerate(metrics_alias):
    with open("results/tables/results_%s.tex" % metric, "w+") as file:
        for id, arg in enumerate(IR_argsorted):
            id += 1
            line = "%d" % (id)
            line_values = []
            line_values = mean_scores[arg, metric_id, :]
            max_value = np.amax(line_values)
            for clf_id, clf_name in enumerate(methods):
                if mean_scores[arg, metric_id, clf_id] == max_value:
                    line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                else:
                    line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
            line += " \\\\"
            print(line, file=file)
            if IR[arg] > 8.6 and IR[arg] < 9.0:
                print("\\hline", file=file)
