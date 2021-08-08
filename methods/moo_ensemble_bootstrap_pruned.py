import numpy as np
import random
from random import randint
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from torch import cdist, from_numpy

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

from methods.optimization_param import OptimizationParam
from utils.diversity import calc_diversity_measures, calc_diversity_measures2


class MooEnsembleSVCbootstrapPruned(BaseEstimator):

    def __init__(self, base_classifier, scale_features=0.75, n_repeats=5, test_size=0.5, objectives=2, p_size=100, predict_decision="ASV", p_minkowski=2, mutation_real="real_pm", mutation_bin="bin_bitflip", crossover_real="real_sbx", crossover_bin="bin_two_point", etac=5, etam=5):

        self.base_classifier = base_classifier
        self.n_repeats = n_repeats
        self.classes = None
        self.test_size = test_size
        self.objectives = objectives
        self.p_size = p_size
        self.scale_features = scale_features
        self.selected_features = []
        self.predict_decision = predict_decision
        self.p_minkowski = p_minkowski
        self.mutation_real = mutation_real
        self.mutation_bin = mutation_bin
        self.crossover_real = crossover_real
        self.crossover_bin = crossover_bin
        self.etac = etac
        self.etam = etam

    def partial_fit(self, X, y, classes=None):
        self.X, self.y = X, y
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(self.y, return_inverse=True)

        n_features = X.shape[1]

        # Mixed variable problem - genetic operators
        mask = ["real", "real"]
        mask.extend(["binary"] * n_features)
        sampling = MixedVariableSampling(mask, {
            "real": get_sampling("real_random"),
            "binary": get_sampling("bin_random")
        })
        crossover = MixedVariableCrossover(mask, {
            "real": get_crossover(self.crossover_real, eta=self.etac),
            "binary": get_crossover(self.crossover_bin)
        })
        mutation = MixedVariableMutation(mask, {
            "real": get_mutation(self.mutation_real, eta=self.etam),
            "binary": get_mutation(self.mutation_bin)
        })

        # Bootstraping - GUMP
        self.roots = []
        # Distances n_samples x n_samples
        # p - parameter Minkowski distance, if p=2 Euclidean distance, if p=1 Manhattan distance, if 0<p<1 it's better for more dimenesions
        self.distances = cdist(from_numpy(self.X), from_numpy(self.X), p=self.p_minkowski).numpy()
        # All samples
        indxs = np.array(list(range(self.X.shape[0])))

        for repeat in range(self.n_repeats):
            self.solutions_bs = None
            if repeat == 0:
                # for the first repeat, it is different way to choose samples
                root = np.random.randint(0, self.X.shape[0]-1)
                self.roots.append(root)
                n2 = np.max(self.distances[root]) - self.distances[root]
                n2 = n2/np.sum(n2)
                # bs_indx - returns indexes of chosen samples
                bs_indx = np.random.choice(indxs, size=int(self.X.shape[0]), replace=True, p=n2)
                bs_X = self.X[bs_indx, :]
                bs_y = self.y[bs_indx]

                minority_samples = sum(1 for i in bs_y if i == 1)
                min_indexes = np.where(y == 1)
                min_indexes = min_indexes[0]
                if minority_samples == 0:
                    maj_indx_a = randint(0, len(bs_y) - 1)
                    maj_indx_b = randint(0, len(bs_y) - 1)
                    min_indx_a = random.choice(min_indexes)
                    min_indx_b = random.choice(min_indexes)
                    bs_y[maj_indx_a] = y[min_indx_a]
                    bs_y[maj_indx_b] = y[min_indx_b]
                elif minority_samples == 1:
                    maj_indexes = np.where(bs_y == 0)
                    maj_indexes = maj_indexes[0]
                    maj_indx = random.choice(maj_indexes)
                    min_indx = random.choice(min_indexes)
                    bs_y[maj_indx] = y[min_indx]

                # Create optimization problem
                problem = OptimizationParam(bs_X, bs_y, test_size=self.test_size, estimator=self.base_classifier, scale_features=self.scale_features, n_features=n_features, objectives=self.objectives)

                algorithm = NSGA2(
                               pop_size=self.p_size,
                               sampling=sampling,
                               crossover=crossover,
                               mutation=mutation,
                               eliminate_duplicates=True)

                res = minimize(
                               problem,
                               algorithm,
                               # termination criterion (n_eval lub n_gen)
                               ('n_eval', 1000),
                               seed=1,
                               verbose=False,
                               save_history=True)

                self.solutions_bs = res.F
                # Unique solutions from the pareto front is a way to prune the ensemble
                unique_solutions, unique_indexes = np.unique(self.solutions_bs, return_index=True, axis=0)
                # self.solutions = unique_solutions
                self.solutions = self.solutions_bs

                # Pruning based on unique solutions of precision and recall from pareto front
                if np.shape(unique_indexes)[0] == 1:
                    self.base_classifier = self.base_classifier.set_params(C=res.X[0, 0], gamma=res.X[0, 1])
                    sf = res.X[0, 2:].tolist()
                    self.selected_features.append(sf)
                    # Train new estimator
                    candidate = clone(self.base_classifier).fit(X[:, sf], y)
                    # Add candidate to the ensemble
                    self.ensemble.append(candidate)
                elif np.shape(unique_indexes)[0] > 1:
                    for indx in unique_indexes:
                        self.base_classifier = self.base_classifier.set_params(C=res.X[0, 0], gamma=res.X[0, 1])
                        sf = res.X[0, 2:].tolist()
                        self.selected_features.append(sf)
                        # Train new estimator
                        candidate = clone(self.base_classifier).fit(X[:, sf], y)
                        # Add candidate to the ensemble
                        self.ensemble.append(candidate)
            else:
                bs_dist = np.mean(self.distances[self.roots], axis=0)
                max_dist = np.argmax(bs_dist)
                self.roots.append(max_dist)
                n2 = np.max(self.distances[max_dist]) - self.distances[max_dist]
                n2 = n2/np.sum(n2)
                bs_indx = np.random.choice(indxs, size=int(self.X.shape[0]), replace=True, p=n2)
                bs_X = self.X[bs_indx, :]
                bs_y = self.y[bs_indx]

                minority_samples = sum(1 for i in bs_y if i == 1)
                min_indexes = np.where(y == 1)
                min_indexes = min_indexes[0]
                if minority_samples == 0:
                    maj_indx_a = randint(0, len(bs_y) - 1)
                    maj_indx_b = randint(0, len(bs_y) - 1)
                    min_indx_a = random.choice(min_indexes)
                    min_indx_b = random.choice(min_indexes)
                    bs_y[maj_indx_a] = y[min_indx_a]
                    bs_y[maj_indx_b] = y[min_indx_b]
                elif minority_samples == 1:
                    maj_indexes = np.where(bs_y == 0)
                    maj_indexes = maj_indexes[0]
                    maj_indx = random.choice(maj_indexes)
                    min_indx = random.choice(min_indexes)
                    bs_y[maj_indx] = y[min_indx]

                # Create optimization problem
                problem = OptimizationParam(bs_X, bs_y, test_size=self.test_size, estimator=self.base_classifier, scale_features=self.scale_features, n_features=n_features, objectives=self.objectives)

                algorithm = NSGA2(
                               pop_size=self.p_size,
                               sampling=sampling,
                               crossover=crossover,
                               mutation=mutation,
                               eliminate_duplicates=True)

                res = minimize(
                               problem,
                               algorithm,
                               ('n_eval', 1000),
                               seed=1,
                               verbose=False,
                               save_history=True)

                self.solutions_bs = res.F
                unique_solutions, unique_indexes = np.unique(self.solutions_bs, return_index=True, axis=0)
                self.solutions = self.solutions_bs
                # self.solutions = np.append(self.solutions, unique_solutions, axis=0)

                # Pruning based on unique solutions of precision and recall from pareto front
                if np.shape(unique_indexes)[0] == 1:
                    self.base_classifier = self.base_classifier.set_params(C=res.X[0, 0], gamma=res.X[0, 1])
                    sf = res.X[0, 2:].tolist()
                    self.selected_features.append(sf)
                    # Train new estimator
                    candidate = clone(self.base_classifier).fit(X[:, sf], y)
                    # Add candidate to the ensemble
                    self.ensemble.append(candidate)
                elif np.shape(unique_indexes)[0] > 1:
                    for indx in unique_indexes:
                        self.base_classifier = self.base_classifier.set_params(C=res.X[0, 0], gamma=res.X[0, 1])
                        sf = res.X[0, 2:].tolist()
                        self.selected_features.append(sf)
                        # Train new estimator
                        candidate = clone(self.base_classifier).fit(X[:, sf], y)
                        # Add candidate to the ensemble
                        self.ensemble.append(candidate)
        return self

    def fit(self, X, y, classes=None):
        self.ensemble = []
        self.partial_fit(X, y, classes)

    def ensemble_support_matrix(self, X):
        # Ensemble support matrix
        return np.array([member_clf.predict_proba(X[:, sf]) for member_clf, sf in zip(self.ensemble, self.selected_features)])

    def predict(self, X):
        # Prediction based on the Average Support Vectors - ONLY THIS!
        if self.predict_decision == "ASV":
            ens_sup_matrix = self.ensemble_support_matrix(X)
            average_support = np.mean(ens_sup_matrix, axis=0)
            prediction = np.argmax(average_support, axis=1)
        # Prediction based on the Majority Voting
        elif self.predict_decision == "MV":
            predictions = np.array([member_clf.predict(X) for member_clf in self.ensemble_])
            prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)

    def calculate_diversity(self):
        if len(self.ensemble) > 1:
            # All measures for whole ensemble
            self.entropy_measure_e, self.k0, self.kw, self.disagreement_measure, self.q_statistic_mean = calc_diversity_measures(self.X, self.y, self.ensemble, self.selected_features, p=0.01)
            # entropy_measure_e: E varies between 0 and 1, where 0 indicates no difference and 1 indicates the highest possible diversity.
            # kw - Kohavi-Wolpert variance
            # Q-statistic: <-1, 1>
            # Q = 0 statistically independent classifiers
            # Q < 0 classifiers commit errors on different objects
            # Q > 0 classifiers recognize the same objects correctly

            return(self.entropy_measure_e, self.kw, self.disagreement_measure, self.q_statistic_mean)

            """
            # k - measurement of interrater agreement
            self.kkk = []
            for sf in self.selected_features:
                # Calculate mean accuracy on training set
                p = np.mean(np.array([accuracy_score(self.y, member_clf.predict(self.X[:, sf])) for repeat, member_clf in enumerate(self.ensemble)]))
                self.k = calc_diversity_measures2(self.X, self.y, self.ensemble, self.selected_features, p, measure="k")
                self.kkk.append(self.k)
            return self.kkk
            """
