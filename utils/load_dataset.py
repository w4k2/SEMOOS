import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/test')


def load_data(filename):
    # Load from file
    with open(filename) as file:
        attr_count = 0
        for line in file:
            if line.startswith("@attribute Class"):
                continue
            elif line.startswith("@attribute"):
                attr_count += 1
            elif line.startswith("@data"):
                break

        df = pd.read_csv(file, header=None)

        features = df.iloc[:, 0:-1]
        # Index of the categorical variables in features
        categorical_id = [key for key in dict(features.dtypes) if dict(features.dtypes)[key] in ['object']]
        # Encoding categorical variables into numbers
        encoder = OrdinalEncoder()
        features[categorical_id] = encoder.fit_transform(features[categorical_id])

        labels = df.iloc[:, -1].values.astype(str)
        classes = np.unique(labels)
        # Enocoding class names into binary
        class_encoder = LabelEncoder()
        labels = class_encoder.fit_transform(labels)
        classes = np.unique(labels)

        return features, labels, classes


def find_datasets(storage=DATASETS_DIR):
    for f_name in os.listdir(storage):
        yield f_name.split('.')[0]


def calc_imbalance_ratio(directories=["9higher_part1", "9higher_part2", "9higher_part3", "9lower"]):
    imbalance_ratios = []
    for dir in directories:
        DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s/' % dir)

        for dataset_id, dataset_name in enumerate(find_datasets(DATASETS_DIR)):
            dataset_path = "/home/joannagrzyb/dev/moo_tune_ensemble/datasets/" + dir + "/" + dataset_name + ".dat"
            X, y, classes = load_data(dataset_path)
            unique, counts = np.unique(y, return_counts=True)

            if len(counts) == 1:
                raise ValueError("Only one class in procesed data.")
            elif counts[0] > counts[1]:
                majority_name = unique[0]
                minority_name = unique[1]
            else:
                majority_name = unique[1]
                minority_name = unique[0]

            minority_ma = np.ma.masked_where(y == minority_name, y)
            minority = X[minority_ma.mask]

            majority_ma = np.ma.masked_where(y == majority_name, y)
            majority = X[majority_ma.mask]

            imbalance_ratio = majority.shape[0]/minority.shape[0]
            imbalance_ratios.append(imbalance_ratio)

    return imbalance_ratios
