import numpy as np
import strlearn as sl
import autograd.numpy as anp
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from pymoo.model.problem import Problem


class OptimizationParam(Problem):
    def __init__(self, X, y, test_size, estimator, scale_features, n_features, n_param=2, objectives=2, random_state=0, feature_names=None):

        self.estimator = estimator
        self.test_size = test_size
        self.objectives = objectives
        self.n_param = n_param
        self.scale_features = scale_features
        self.n_features = n_features

        self.X = X
        self.y = y
        if self.test_size != 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, stratify=self.y)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = np.copy(self.X), np.copy(self.y), np.copy(self.X), np.copy(self.y)

        # Lower and upper bounds for x - 1d array with length equal to number of variable
        xl_real = [1E6, 1E-7]
        xl_binary = [0] * n_features
        xl = np.hstack([xl_real, xl_binary])
        xu_real = [1E9, 1E-4]
        xu_binary = [1] * n_features
        xu = np.hstack([xu_real, xu_binary])
        n_variable = self.n_param + self.n_features

        super().__init__(n_var=n_variable, n_obj=objectives,
                         n_constr=1, elementwise_evaluation=True, xl=xl, xu=xu)

    # x: a two dimensional matrix where each row is a point to evaluate and each column a variable
    def validation(self, x):
        C = x[0]
        gamma = x[1]
        selected_features = x[2:]
        selected_features = selected_features.tolist()

        # If at least one element in x are True
        if True in selected_features:
            clf = None
            clf = clone(self.estimator.set_params(C=C, gamma=gamma))
            clf.fit(self.X_train[:, selected_features], self.y_train)
            y_pred = clf.predict(self.X_test[:, selected_features])
            metrics = [sl.metrics.precision(self.y_test, y_pred), sl.metrics.recall(self.y_test, y_pred)]
        else:
            metrics = [0, 0]
        return metrics

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)
        # print(scores)

        # Function F is always minimize, but the minus sign (-) before F means maximize
        f1 = -1 * scores[0]
        f2 = -1 * scores[1]
        out["F"] = anp.column_stack(np.array([f1, f2]))

        # Function constraint to select specific numbers of features:
        number = int((1 - self.scale_features) * self.n_features)
        out["G"] = (self.n_features - np.sum(x[2:]) - number) ** 2
