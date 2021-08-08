import os
import numpy as np
import matplotlib.pyplot as plt
from utils.load_dataset import find_datasets


# Plot pareto front scatter function
def scatter_pareto_chart(DATASETS_DIR, n_folds, experiment_name, methods, methods_alias):
    n_rows_p = 1000
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        print(dataset)
        for clf_id, clf_name in enumerate(methods):
            for fold_id in range(n_folds):
                solutions = []
                for sol_id in range(n_rows_p):
                    try:
                        filename_pareto = "results/%s/pareto_raw/%s/%s/fold%d/sol%d.csv" % (experiment_name, dataset, clf_name, fold_id, sol_id)
                        solution = np.genfromtxt(filename_pareto, dtype=np.float32)
                        solution = solution.tolist()
                        solution[0] = solution[0] * (-1)
                        solution[1] = solution[1] * (-1)
                        solutions.append(solution)
                    except IOError:
                        pass
                if solutions:
                    filename_pareto_chart = "results/%s/pareto_plots/%s/%s/pareto_%s_%s_fold%d" % (experiment_name, dataset, clf_name, dataset, clf_name, fold_id)
                    if not os.path.exists("results/%s/pareto_plots/%s/%s/" % (experiment_name, dataset, clf_name)):
                        os.makedirs("results/%s/pareto_plots/%s/%s/" % (experiment_name, dataset, clf_name))
                    x = []
                    y = []
                    for solution in solutions:
                        x.append(solution[0])
                        y.append(solution[1])
                    x = np.array(x)
                    y = np.array(y)
                    plt.grid(True, color="silver", linestyle=":", axis='both')
                    plt.scatter(x, y, color='black')
                    plt.title("Objective Space", fontsize=12)
                    plt.xlabel('Precision', fontsize=12)
                    plt.ylabel('Recall', fontsize=12)
                    plt.gcf().set_size_inches(6, 3)
                    plt.savefig(filename_pareto_chart+".png", bbox_inches='tight')
                    plt.savefig(filename_pareto_chart+".eps", format='eps', bbox_inches='tight')
                    plt.clf()
                    plt.close()


# Plot scatter of pareto front solutions and all methods
def scatter_plot(datasets, n_folds, experiment_name, methods, raw_data):
    n_rows_p = 1000
    for dataset_id, dataset in enumerate(datasets):
        print(dataset)
        for fold_id in range(n_folds):
            solutions_semoos = []
            solutions_semoosb = []
            solutions_semoosbp = []
            for sol_id in range(n_rows_p):
                try:
                    filename_pareto_semoos = "results/%s/pareto_raw/%s/MooEnsembleSVC/fold%d/sol%d.csv" % (experiment_name, dataset, fold_id, sol_id)
                    solution_semoos = np.genfromtxt(filename_pareto_semoos, dtype=np.float32)
                    solution_semoos = solution_semoos.tolist()
                    solution_semoos[0] = solution_semoos[0] * (-1)
                    solution_semoos[1] = solution_semoos[1] * (-1)
                    solutions_semoos.append(solution_semoos)
                except IOError:
                    pass
                try:
                    filename_pareto_semoosb = "results/%s/pareto_raw/%s/MooEnsembleSVCbootstrap/fold%d/sol%d.csv" % (experiment_name, dataset, fold_id, sol_id)
                    solution_semoosb = np.genfromtxt(filename_pareto_semoosb, dtype=np.float32)
                    solution_semoosb = solution_semoosb.tolist()
                    solution_semoosb[0] = solution_semoosb[0] * (-1)
                    solution_semoosb[1] = solution_semoosb[1] * (-1)
                    solutions_semoosb.append(solution_semoosb)
                except IOError:
                    pass
                try:
                    filename_pareto_semoosbp = "results/%s/pareto_raw/%s/MooEnsembleSVCbootstrapPruned/fold%d/sol%d.csv" % (experiment_name, dataset, fold_id, sol_id)
                    solution_semoosbp = np.genfromtxt(filename_pareto_semoosbp, dtype=np.float32)
                    solution_semoosbp = solution_semoosbp.tolist()
                    solution_semoosbp[0] = solution_semoosbp[0] * (-1)
                    solution_semoosbp[1] = solution_semoosbp[1] * (-1)
                    solutions_semoosbp.append(solution_semoosbp)
                except IOError:
                    pass
            if solutions_semoos and solutions_semoosb and solutions_semoosbp:
                filename_pareto_chart = "results/%s/scatter_plots/%s/scatter_%s_fold%d" % (experiment_name, dataset, dataset, fold_id)
                if not os.path.exists("results/%s/scatter_plots/%s/" % (experiment_name, dataset)):
                    os.makedirs("results/%s/scatter_plots/%s/" % (experiment_name, dataset))

                semoos_x = []
                semoos_y = []
                for solution in solutions_semoos:
                    semoos_x.append(solution[0])
                    semoos_y.append(solution[1])
                semoos_x = np.array(semoos_x)
                semoos_y = np.array(semoos_y)
                semoosb_x = []
                semoosb_y = []
                for solution in solutions_semoosb:
                    semoosb_x.append(solution[0])
                    semoosb_y.append(solution[1])
                semoosb_x = np.array(semoosb_x)
                semoosb_y = np.array(semoosb_y)
                semoosbp_x = []
                semoosbp_y = []
                for solution in solutions_semoosbp:
                    semoosbp_x.append(solution[0])
                    semoosbp_y.append(solution[1])
                semoosbp_x = np.array(semoosbp_x)
                semoosbp_y = np.array(semoosbp_y)

                plt.grid(True, color="silver", linestyle=":", axis='both')

                # SEMOOS pareto
                plt.scatter(semoos_x, semoos_y, color='darkgray', marker="o", label="SEMOOS PF")
                # SEMOOS one
                # Precision
                semoos_p = raw_data[dataset_id, 6, 0, fold_id]
                # Recall
                semoos_r = raw_data[dataset_id, 4, 0, fold_id]
                plt.scatter(semoos_p, semoos_r, color='black', marker="o", label="SEMOOS")
                # SEMOOSb pareto
                plt.scatter(semoosb_x, semoosb_y, color='tab:cyan', marker="x", label="SEMOOSb PF")
                # SEMOOSb one
                semoosb_p = raw_data[dataset_id, 6, 1, fold_id]
                semoosb_r = raw_data[dataset_id, 4, 1, fold_id]
                plt.scatter(semoosb_p, semoosb_r, color='darkslategray', marker="X", label="SEMOOSb")
                # # SEMOOSbp pareto
                plt.scatter(semoosbp_x, semoosbp_y, color='peachpuff', marker="+", label="SEMOOSbp PF")
                # SEMOOSbp one
                semoosbp_p = raw_data[dataset_id, 6, 2, fold_id]
                semoosbp_r = raw_data[dataset_id, 4, 2, fold_id]
                plt.scatter(semoosbp_p, semoosbp_r, color='peru', marker="P", label="SEMOOSbp")
                # RS
                plt.scatter(raw_data[dataset_id, 6, 3, fold_id], raw_data[dataset_id, 4, 3, fold_id], color='tab:blue', marker="v", label="RS")
                # SVM
                plt.scatter(raw_data[dataset_id, 6, 4, fold_id], raw_data[dataset_id, 4, 4, fold_id], color='tab:red', marker="^", label="SVM")
                # FS
                plt.scatter(raw_data[dataset_id, 6, 5, fold_id], raw_data[dataset_id, 4, 5, fold_id], color='tab:purple', marker="<", label="FS")
                # FSIRSVM
                plt.scatter(raw_data[dataset_id, 6, 6, fold_id], raw_data[dataset_id, 4, 6, fold_id], color='tab:pink', marker=">", label="FSIRSVM")

                # plt.title("Objective Space", fontsize=12)
                plt.xlabel('Precision', fontsize=12)
                plt.ylabel('Recall', fontsize=12)
                plt.xlim([0, 1.1])
                plt.ylim([0, 1.1])
                plt.legend(loc="best")
                plt.gcf().set_size_inches(9, 6)
                plt.savefig(filename_pareto_chart+".png", bbox_inches='tight')
                plt.savefig(filename_pareto_chart+".eps", format='eps', bbox_inches='tight')
                plt.clf()
                plt.close()


def diversity_bar_plot(diversity, diversity_measures, methods_ens_alias, experiment_name):
    for metric_id, metric in enumerate(diversity_measures):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(methods_ens_alias, diversity[:, metric_id], width=0.4, color=["#2F3441", "#877669", "#A6B1A2", "#E2B3A9"])

        plt.grid(True, color="silver", linestyle=":", axis='y', which='major')
        plt.ylabel(f"{metric}", fontsize=14)
        plt.xlabel("Methods", fontsize=14)
        plt.gcf().set_size_inches(3, 3)
        # Save plot
        filename = "results/experiment_server/%s/diversity_plot/diversity_bar_plot_%s_%s" % (experiment_name, metric, experiment_name)
        if not os.path.exists("results/experiment_server/%s/diversity_plot/" % (experiment_name)):
            os.makedirs("results/experiment_server/%s/diversity_plot/" % (experiment_name))
        plt.savefig(filename+".png", bbox_inches='tight')
        plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
        plt.clf()
        plt.close()
