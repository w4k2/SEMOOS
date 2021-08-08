import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import chi2

from methods.moo_ensemble import MooEnsembleSVC
from methods.moo_ensemble_bootstrap import MooEnsembleSVCbootstrap
from methods.moo_ensemble_bootstrap_pruned import MooEnsembleSVCbootstrapPruned
from methods.random_subspace_ensemble import RandomSubspaceEnsemble
from methods.feature_selection_clf import FeatueSelectionClf
from utils.load_dataset import find_datasets
from utils.plots import scatter_pareto_chart, scatter_plot
from utils.wilcoxon_ranking import pairs_metrics_multi


DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/9higher_part2')
n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))

base_estimator = {'SVM': SVC(probability=True)}
methods = {
    "MooEnsembleSVC": MooEnsembleSVC(base_classifier=base_estimator),
    "MooEnsembleSVCbootstrap": MooEnsembleSVCbootstrap(base_classifier=base_estimator),
    "MooEnsembleSVCbootstrapPruned": MooEnsembleSVCbootstrapPruned(base_classifier=base_estimator),
    "RandomSubspace": RandomSubspaceEnsemble(base_classifier=base_estimator),
    "SVM": SVC(),
    "FS": FeatueSelectionClf(base_estimator, chi2),
    "FSIRSVM": 0
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
data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))

# Load data from file
datasets = []
for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    datasets.append(dataset)
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            try:
                filename = "results/experiment_server/experiment2_9higher_part2/raw_results/%s/%s/%s.csv" % (metric, dataset, clf_name)
                scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                data_np[dataset_id, metric_id, clf_id] = scores
                mean_score = np.mean(scores)
                mean_scores[dataset_id, metric_id, clf_id] = mean_score
                std = np.std(scores)
                stds[dataset_id, metric_id, clf_id] = std
                print(dataset, clf_name, metric, mean_score, std)
            except:
                print("Error loading data!")


# Plotting
# Bar chart function
def horizontal_bar_chart():
    for metric_id, metric in enumerate(metrics_alias):
        data = {}
        stds_data = {}
        for clf_id, clf_name in enumerate(methods_alias):
            plot_data = []
            stds_errors = []
            for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
                plot_data.append(mean_scores[dataset_id, metric_id, clf_id])
                stds_errors.append(stds[dataset_id, metric_id, clf_id])
            data[clf_name] = plot_data
            stds_data[clf_name] = stds_errors
        df = pd.DataFrame(data, columns=methods_alias, index=datasets)
        print(df)
        df = df.sort_index()

        ax = df.plot.barh(xerr=stds_data)
        ax.invert_yaxis()
        plt.ylabel("Datasets")
        plt.xlabel("Score")
        plt.title(f"Metric: {metric}")
        plt.legend(loc='best')
        plt.grid(True, color="silver", linestyle=":", axis='both', which='both')
        plt.gcf().set_size_inches(9, 14)
        # Save plot
        filename = "results/experiment_server/experiment2_9higher_part2/plot_bar/ex2_bar_%s" % (metric)
        if not os.path.exists("results/experiment_server/experiment2_9higher_part2/plot_bar/"):
            os.makedirs("results/experiment_server/experiment2_9higher_part2/plot_bar/")
        plt.savefig(filename+".png", bbox_inches='tight')
        plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
        plt.clf()
        plt.close()


# Plotting bar chart
# horizontal_bar_chart()

# Plot pareto front scatter
# scatter_pareto_chart(DATASETS_DIR=DATASETS_DIR, n_folds=n_folds, experiment_name="experiment_server/experiment2_9higher_part2")

# Plot scatter with all methods
scatter_plot(datasets=datasets, n_folds=n_folds, experiment_name="experiment_server/experiment2_9higher_part2", methods=methods, raw_data=data_np)

# Wilcoxon ranking - statistic test for methods: SEMOOS and SEMOOSb
# pairs_metrics_multi(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment2_9higher_part2", dataset_names=datasets, metrics=metrics_alias, filename="ex2_ranking_plot", ref_method=methods_alias[0])
#
# pairs_metrics_multi(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment2_9higher_part2", dataset_names=datasets, metrics=metrics_alias, filename="ex2_ranking_plot", ref_method=methods_alias[1])
