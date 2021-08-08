from sklearn.base import BaseEstimator, clone
import numpy as np
from utils.diversity import calc_diversity_measures, calc_diversity_measures2


class RandomSubspaceEnsemble(BaseEstimator):
    def __init__(self, base_classifier, n_members=100, subspace_size=3):
        self.base_classifier = base_classifier
        self.n_members = n_members
        self.subspace_size = subspace_size

    def fit(self, X, y):
        self.X, self.y = X, y
        self.n_features = X.shape[1]
        self.subspaces = np.random.randint(
            self.n_features, size=(self.n_members, self.subspace_size)
        )

        self.ensemble = []
        for subspace in self.subspaces:
            clf = clone(self.base_classifier).fit(X[:, subspace], y)
            self.ensemble.append(clf)

        return self

    def predict_proba(self, X):
        esm = np.mean(
            np.array(
                [
                    clf.predict_proba(X[:, self.subspaces[i]])
                    for i, clf in enumerate(self.ensemble)
                ]
            ),
            axis=0,
        )

        return esm

    def predict(self, X):
        pp = self.predict_proba(X)

        y_pred = np.argmax(pp, axis=1)

        return y_pred

    def calculate_diversity(self):
        if len(self.ensemble) > 1:
            # All measures for whole ensemble
            self.entropy_measure_e, self.k0, self.kw, self.disagreement_measure, self.q_statistic_mean = calc_diversity_measures(self.X, self.y, self.ensemble, self.subspaces, p=0.01)
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
