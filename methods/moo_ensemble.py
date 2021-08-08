import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from scipy.stats import mode

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

from methods.optimization_param import OptimizationParam
from utils.diversity import calc_diversity_measures, calc_diversity_measures2


class MooEnsembleSVC(BaseEstimator):

    def __init__(self, base_classifier, scale_features=0.75, n_classifiers=10, test_size=0.5, objectives=2, p_size=100, predict_decision="ASV", p_minkowski=2, mutation_real="real_pm", mutation_bin="bin_bitflip", crossover_real="real_sbx", crossover_bin="bin_two_point", etac=5, etam=5):

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
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

        # Create optimization problem
        problem = OptimizationParam(X, y, test_size=self.test_size, estimator=self.base_classifier, scale_features=self.scale_features, n_features=n_features, objectives=self.objectives)

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

        # F returns all Pareto front solutions in form [-precision, -recall]
        self.solutions = res.F

        # X returns values of hyperparameter C, gamma and binary vector of selected features
        for result_opt in res.X:
            self.base_classifier = self.base_classifier.set_params(C=result_opt[0], gamma=result_opt[1])
            sf = result_opt[2:].tolist()
            self.selected_features.append(sf)
            # Train new estimator
            candidate = clone(self.base_classifier).fit(X[:, sf], y)
            # Add candidate to the ensemble
            self.ensemble.append(candidate)

        """
        # Pruning based on balanced_accuracy_score
        ensemble_size = len(self.ensemble)
        if ensemble_size > self.n_classifiers:
            bac_array = []
            for clf_id, clf in enumerate(self.ensemble):
                y_pred = clf.predict(X[:, self.selected_features[clf_id]])
                bac = sl.metrics.balanced_accuracy_score(y, y_pred)
                bac_array.append(bac)
            bac_arg_sorted = np.argsort(bac_array)
            self.ensemble_arr = np.array(self.ensemble)
            self.ensemble_arr = self.ensemble_arr[bac_arg_sorted[(len(bac_array)-self.n_classifiers):]]
            self.ensemble = self.ensemble_arr.tolist()
        """

        return self

    def fit(self, X, y, classes=None):
        self.ensemble = []
        self.partial_fit(X, y, classes)

    def ensemble_support_matrix(self, X):
        # Ensemble support matrix
        return np.array([member_clf.predict_proba(X[:, sf]) for member_clf, sf in zip(self.ensemble, self.selected_features)])

    def predict(self, X):
        # Prediction based on the Average Support Vectors - to wybierz!
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
                p = np.mean(np.array([accuracy_score(self.y, member_clf.predict(self.X[:, sf])) for clf_id, member_clf in enumerate(self.ensemble)]))
                self.k = calc_diversity_measures2(self.X, self.y, self.ensemble, self.selected_features, p, measure="k")
                self.kkk.append(self.k)
            return self.kkk
            """
