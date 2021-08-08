import os
import numpy as np
import warnings
from load_dataset import load_data, find_datasets, calc_imbalance_ratio


warnings.filterwarnings("ignore")

directories = ["9lower", "9higher_part1", "9higher_part2", "9higher_part3"]

n_datasets = 0
datasets = []
for dir_id, dir in enumerate(directories):
    DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s/' % dir)
    n_datasets += len(list(enumerate(find_datasets(DATASETS_DIR))))
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        datasets.append(dataset)

X_all = []
y_all = []
for dir in directories:
    DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s/' % dir)
    for dataset_id, dataset_name in enumerate(datasets):
        dataset_path = "/home/joannagrzyb/dev/moo_tune_ensemble/datasets/" + dir + "/" + dataset_name + ".dat"
        if not os.path.isfile(dataset_path):
            # print("File not exist - %s" % filename)
            continue
        X, y, classes = load_data(dataset_path)
        X_all.append(X)
        y_all.append(y)

IR = calc_imbalance_ratio(directories=directories)
IR_argsorted = np.argsort(IR)

with open("tables/datasets.tex", "w+") as file:
    for id, arg in enumerate(IR_argsorted):
        id += 1
        number_of_features = len(X_all[arg].columns)
        number_of_objects = len(y_all[arg])
        dataset_name = datasets[arg].replace("_", "\\_")
        print("%d & \\emph{%s} & %0.2f & %d & %d \\\\" % (id, dataset_name, IR[arg], number_of_objects, number_of_features), file=file)
        if IR[arg] > 8.6 and IR[arg] < 9.0:
            print("\\hline", file=file)
