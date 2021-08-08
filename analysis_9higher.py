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
from utils.plots import scatter_pareto_chart, diversity_bar_plot
from utils.wilcoxon_ranking import pairs_metrics_multi
from utils.wilcoxon_ranking_grid import pairs_metrics_multi_grid
from utils.wilcoxon_ranking_grid_all import pairs_metrics_multi_grid_all


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
diversity_measures = ["Entropy", "KW", "Disagreement", "Q statistic"]

n_splits = 2
n_repeats = 5
n_folds = n_splits * n_repeats
n_methods = len(methods_alias) * len(base_estimator)
n_metrics = len(metrics_alias)

# Load data from file
n_datasets = 0
datasets = []
directories = ["9higher_part1", "9higher_part2", "9higher_part3"]
for dir_id, dir in enumerate(directories):
    DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s/' % dir)
    n_datasets += len(list(enumerate(find_datasets(DATASETS_DIR))))
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        datasets.append(dataset)

data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))
diversity = np.zeros((n_datasets, 4, n_folds, len(diversity_measures)))

for dir_id, dir in enumerate(directories):
    dir_id += 1
    for dataset_id, dataset in enumerate(datasets):
        for clf_id, clf_name in enumerate(methods):
            for metric_id, metric in enumerate(metrics_alias):
                try:
                    filename = "results/experiment_server/experiment%d_%s/raw_results/%s/%s/%s.csv" % (dir_id, dir, metric, dataset, clf_name)
                    if not os.path.isfile(filename):
                        # print("File not exist - %s" % filename)
                        continue
                    scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    data_np[dataset_id, metric_id, clf_id] = scores
                    mean_score = np.mean(scores)
                    mean_scores[dataset_id, metric_id, clf_id] = mean_score
                    std = np.std(scores)
                    stds[dataset_id, metric_id, clf_id] = std
                except:
                    print("Error loading data!", dataset, clf_name, metric)

                for div_measure_id, div_measure in enumerate(diversity_measures):
                    try:
                        filename = "results/experiment_server/experiment%d_%s/diversity_results/%s/%s.csv" % (dir_id, dir, dataset, clf_name)
                        if not os.path.isfile(filename):
                            # print("File not exist - %s" % filename)
                            continue
                        diversity_raw = np.genfromtxt(filename, delimiter=' ', dtype=np.float32)
                        if np.isnan(diversity_raw).all():
                            pass
                        else:
                            diversity_raw = np.nan_to_num(diversity_raw)
                            diversity[dataset_id, clf_id] = diversity_raw
                    except:
                        print("Error loading diversity data!", dataset, clf_name, div_measure)

diversity_m = np.mean(diversity, axis=2)
diversity_mean = np.mean(diversity_m, axis=0)
print(diversity_mean)

# print(mean_scores)
# print(datasets, len(datasets))

# Plotting
# Bar chart function
def horizontal_bar_chart():
    for metric_id, metric in enumerate(metrics_alias):
        data = {}
        stds_data = {}
        for clf_id, clf_name in enumerate(methods_alias):
            plot_data = []
            stds_errors = []
            for dataset_id, dataset in enumerate(datasets):
                plot_data.append(mean_scores[dataset_id, metric_id, clf_id])
                stds_errors.append(stds[dataset_id, metric_id, clf_id])
            data[clf_name] = plot_data
            stds_data[clf_name] = stds_errors
        # print(len(plot_data))
        # print(data)
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
        plt.gcf().set_size_inches(8, 30)
        # Save plot
        filename = "results/experiment_server/experiment_9higher/plot_bar/bar_%s" % (metric)
        if not os.path.exists("results/experiment_server/experiment_9higher/plot_bar/"):
            os.makedirs("results/experiment_server/experiment_9higher/plot_bar/")
        plt.savefig(filename+".png", bbox_inches='tight')
        plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
        plt.clf()
        plt.close()


# Plotting bar chart
# horizontal_bar_chart()

# Wilcoxon ranking horizontally- statistic test for methods: SEMOOS
# pairs_metrics_multi(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment_9higher", dataset_names=datasets, metrics=metrics_alias, filename="ex9h_ranking_plot", ref_method=methods_alias[0])
# # SEMOOSb
# pairs_metrics_multi(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment_9higher", dataset_names=datasets, metrics=metrics_alias, filename="ex9h_ranking_plot", ref_method=methods_alias[1])
# # SEMOOSbp
# pairs_metrics_multi(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment_9higher", dataset_names=datasets, metrics=metrics_alias, filename="ex9h_ranking_plot", ref_method=methods_alias[2])

# Wilcoxon ranking grid - statistic test for methods: SEMOOS, SEMOOSb, SEMOOSbp and all metrics
# UNCOMMENT ONLY methods SEMOOS, SEMOOSb, SEMOOSbp
# pairs_metrics_multi_grid(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment_9higher", dataset_names=datasets, metrics=metrics_alias, filename="ex9h_ranking_plot_grid_variants", ref_methods=methods_alias[0:3], offset=-75)

# Wilcoxon ranking grid - statistic test for all methods vs: SEMOOS, SEMOOSb, SEMOOSbp and all metrics
pairs_metrics_multi_grid_all(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment_9higher", dataset_names=datasets, metrics=metrics_alias, filename="ex9h_ranking_plot_grid_all", ref_methods=methods_alias[0:3], offset=-75)


# Diveersity bar Plotting
# diversity_bar_plot(diversity_mean, diversity_measures, methods_alias[:4], experiment_name="experiment_9higher")
