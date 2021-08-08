from scipy import stats
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import sqrt, ceil


def pairs_metrics_multi(method_names, data_np, experiment_name, dataset_names, metrics, filename, ref_method, treshold=0.5):
    # Load data
    data = {}
    for dataset_id, dataset_name in enumerate(dataset_names):
        for method_id, method_name in enumerate(method_names):
            for metric_id, metric in enumerate(metrics):
                try:
                    if metric == "Gmean2" or metric == "F1score":
                        continue
                    else:
                        data[(method_name, dataset_name, metric)] = data_np[dataset_id, metric_id, method_id]
                except:
                    print("None is ", method_name, dataset_name, metric)
                    data[(method_name, dataset_name, metric)] = None
                    print(data[(method_name, dataset_name, metric)])

    # Remove unnecessary metrics
    if "Gmean2" in metrics:
        metrics.remove("Gmean2")
    if "F1score" in metrics:
        metrics.remove("F1score")

    plt.rc('ytick', labelsize=12)
    fig, axes = plt.subplots(1, len(metrics))
    fig.subplots_adjust(wspace=0.6)

    # Init/clear ranks
    for index_j, metric in enumerate(metrics):
        ranking = {}
        for method_name in method_names:
            ranking[method_name] = {
                "win": 0, "lose": 0, "tie": 0, "error": 0}

        # Pair tests
        for dataset in tqdm(dataset_names, "Rank %s" % (metric)):
            method_1 = ref_method
            for j, method_2 in enumerate(method_names):
                if method_1 == method_2:
                    continue
                try:
                    statistic, p_value = stats.ranksums(data[(method_1, dataset, metric)], data[(
                        method_2, dataset, metric)])
                    if p_value < treshold:
                        if statistic > 0:
                            ranking[method_2]["win"] += 1
                        else:
                            ranking[method_2]["lose"] += 1
                    else:
                        ranking[method_2]["tie"] += 1
                except:
                    ranking[method_2]["error"] += 1
                    print("Exception", method_1, method_2,
                          dataset, metric)

        # Count ranks
        rank_win = []
        rank_tie = []
        rank_lose = []
        rank_error = []

        method_names_c = [x for x in method_names if x != ref_method]
        for method_name in method_names_c:
            rank_win.append(ranking[method_name]['win'])
            rank_tie.append(ranking[method_name]['tie'])
            rank_lose.append(ranking[method_name]['lose'])
            try:
                rank_error.append(ranking[method_name]['error'])
            except Exception:
                pass

        rank_win.reverse()
        rank_tie.reverse()
        rank_lose.reverse()
        rank_error.reverse()

        rank_win = np.array(rank_win)
        rank_tie = np.array(rank_tie)
        rank_lose = np.array(rank_lose)
        rank_error = np.array(rank_error)
        ma = method_names_c.copy()
        ma.reverse()

        # Plotting
        try:
            axes[index_j].barh(
                ma, rank_error, color="green", height=0.9)
            axes[index_j].barh(
                ma, rank_win, left=rank_error, color="green", height=0.9)
            axes[index_j].barh(
                ma, rank_tie, left=rank_error + rank_win, color="gold", height=0.9)
            axes[index_j].barh(
                ma, rank_lose, left=rank_error + rank_win + rank_tie, color="crimson", height=0.9)
            axes[index_j].set_xlim([0, len(dataset_names)])
        except Exception:
            axes[index_j].barh(
                ma, rank_win, color="green", height=0.9)
            axes[index_j].barh(
                ma, rank_tie, left=rank_win, color="gold", height=0.9)
            axes[index_j].barh(
                ma, rank_lose, left=rank_win + rank_tie, color="crimson", height=0.9)
            axes[index_j].set_xlim([0, len(dataset_names)])

        # Calculate and plot critical difference
        N_of_streams = len(dataset_names)
        critical_difference = ceil(
            N_of_streams / 2 + 1.96 * sqrt(N_of_streams) / 2)
        if len(dataset_names) < 25:
            axes[index_j].axvline(
                critical_difference, 0, 1, linestyle="--", linewidth=3, color="black")
        else:
            axes[index_j].axvline(
                critical_difference, 0, 1, linestyle="--", linewidth=3, color="red")

    for j, metric in enumerate(metrics):
        axes[j].set_title(metric.upper(), fontsize=18)

    if not os.path.exists("results/%s/ranking/" % (experiment_name)):
        os.makedirs("results/%s/ranking/" % (experiment_name))
    plt.gcf().set_size_inches(15, 2)
    filepath = "results/%s/ranking/%s_%s" % (experiment_name, filename, ref_method)
    plt.savefig(filepath + ".png", bbox_inches='tight')
    plt.savefig(filepath + ".eps", format='eps', bbox_inches='tight')
    plt.clf()
