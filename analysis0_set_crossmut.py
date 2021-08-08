import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from methods.moo_ensemble_bootstrap import MooEnsembleSVCbootstrap
from utils.load_dataset import find_datasets


DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/set_params')
n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))

base_estimator = {'SVM': SVC(probability=True)}

methods = {
    "MooEnsembleSVCbootstrap_ce2_me2": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=2),
    "MooEnsembleSVCbootstrap_ce5_me2": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=5, etam=2),
    "MooEnsembleSVCbootstrap_ce10_me2": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=10, etam=2),
    "MooEnsembleSVCbootstrap_ce20_me2": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=20, etam=2),
    "MooEnsembleSVCbootstrap_ce2_me5": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=5),
    "MooEnsembleSVCbootstrap_ce5_me5": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=5, etam=5),
    "MooEnsembleSVCbootstrap_ce10_me5": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=10, etam=5),
    "MooEnsembleSVCbootstrap_ce20_me5": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=20, etam=5),
    "MooEnsembleSVCbootstrap_ce2_me10": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=10),
    "MooEnsembleSVCbootstrap_ce5_me10": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=5, etam=10),
    "MooEnsembleSVCbootstrap_ce10_me10": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=10, etam=10),
    "MooEnsembleSVCbootstrap_ce20_me10": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=20, etam=10),
    "MooEnsembleSVCbootstrap_ce2_me20": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=2, etam=20),
    "MooEnsembleSVCbootstrap_ce5_me20": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=5, etam=20),
    "MooEnsembleSVCbootstrap_ce10_me20": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=10, etam=20),
    "MooEnsembleSVCbootstrap_ce20_me20": MooEnsembleSVCbootstrap(base_classifier=base_estimator, etac=20, etam=20),
    }

metrics_alias = ["BAC", "Gmean", "Gmean2", "F1score", "Recall", "Specificity", "Precision"]

n_splits = 2
n_repeats = 5
n_folds = n_splits * n_repeats
n_methods = len(methods) * len(base_estimator)
n_metrics = len(metrics_alias)
data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
mean_scores_fold = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))

# Load data from file
datasets = []
for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    datasets.append(dataset)
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            try:
                filename = "results/experiment0_set_crossmut/raw_results/%s/%s/%s.csv" % (metric, dataset, clf_name)
                scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                data_np[dataset_id, metric_id, clf_id] = scores
                mean_score = np.mean(scores)
                mean_scores_fold[dataset_id, metric_id, clf_id] = mean_score
                std = np.std(scores)
                stds[dataset_id, metric_id, clf_id] = std
                # print(dataset, clf_name, metric, mean_score, std)
            except:
                print("Error loading data!")

mean_scores_ds = np.mean(mean_scores_fold, axis=0)
etas = [2, 5, 10, 20]


def save_grid_results(param_1, param_2, metrics_array, dataset_name, metrics_alias):
    for metric_id, metric in enumerate(metrics_alias):

        filename = ("%s_%s_%s_" % (dataset_name, param_1, param_2)) + metric
        filepath = "results/experiment0_set_crossmut/grid/values/%s" % filename

        if not os.path.exists("results/experiment0_set_crossmut/grid/values/"):
            os.makedirs("results/experiment0_set_crossmut/grid/values/")

        np.save(filepath, metrics_array[metric_id, :, :])
        np.savetxt(filepath + ".csv",
                   metrics_array[metric_id, :, :],
                   delimiter=";",
                   fmt="%0.3f")


def plot_grid_results(param_1, param_2, metrics_array, dataset_name, metrics_alias, etas):
    for metric_id, metric in enumerate(metrics_alias):

        fig, ax = plt.subplots(figsize=(3, 3))
        mean = np.mean(metrics_array[metric_id, :, :])
        max = np.max(metrics_array[metric_id, :, :])
        ax.imshow(metrics_array[metric_id, :, :], cmap='Greys')
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.title(metric)

        labels = etas
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels[::-1])
        for i in range(len(labels)):
            for j in range(len(labels)):
                if metrics_array[metric_id, i, j] > mean:
                    if metrics_array[metric_id, i, j] == max:
                        ax.text(j, i,
                                '%.3f' % metrics_array[metric_id, i, j],
                                ha="center",
                                va="center",
                                color="w",
                                fontweight="bold",
                                fontsize="11")
                        continue
                    ax.text(j, i,
                            '%.3f' % metrics_array[metric_id, i, j],
                            ha="center",
                            va="center",
                            color="w")
                else:
                    ax.text(j, i,
                            '%.3f' % metrics_array[metric_id, i, j],
                            ha="center",
                            va="center",
                            color="black")

        if not os.path.exists("results/experiment0_set_crossmut/grid/plots/"):
            os.makedirs("results/experiment0_set_crossmut/grid/plots/")

        filename = ("%s_%s_%s_" % (dataset_name, param_1, param_2)) + metric

        plt.tight_layout()
        plt.savefig("results/experiment0_set_crossmut/grid/plots/%s.png" % filename)
        plt.savefig("results/experiment0_set_crossmut/grid/plots/%s.eps" % filename)

        filepath = "results/experiment0_set_crossmut/grid/values/%s" % filename
        if not os.path.exists("results/experiment0_set_crossmut/grid/values/"):
            os.makedirs("results/experiment0_set_crossmut/grid/values/")

        np.save(filepath, metrics_array[metric_id, :, :])
        np.savetxt(filepath + ".csv",
                   metrics_array[metric_id, :, :],
                   delimiter=";",
                   fmt="%0.3f")


# Average resuls dor 4 datasets
metrics_array = np.zeros((len(metrics_alias), len(etas), len(etas)))
for metric_id, metric in enumerate(metrics_alias):
    metrics_array[metric_id] = np.reshape(mean_scores_ds[metric_id, :], (4, 4))

metrics_array_r = np.flip(metrics_array, 1)
print(metrics_array_r)
save_grid_results("Eta_c", "Eta_m", metrics_array_r, "avg", metrics_alias)

plot_grid_results("Eta_c", "Eta_m", metrics_array_r, "avg", metrics_alias, etas)

# Results for each dataset
data_array = np.zeros((len(datasets), len(metrics_alias), len(etas), len(etas)))
for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    for metric_id, metric in enumerate(metrics_alias):
        data_array[dataset_id, metric_id] = np.reshape(mean_scores_fold[dataset_id, metric_id, :], (len(etas), len(etas)))

data_array_r = np.flip(data_array, 2)
print(data_array_r)

for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    save_grid_results("Eta_c", "Eta_m", data_array_r[dataset_id], dataset, metrics_alias)

    plot_grid_results("Eta_c", "Eta_m", data_array_r[dataset_id], dataset, metrics_alias, etas)
