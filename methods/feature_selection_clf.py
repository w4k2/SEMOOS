import warnings
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.feature_selection import SelectKBest

warnings.simplefilter(action='ignore', category=FutureWarning)


class FeatueSelectionClf(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, score_function, scale_features=0.75):
        self.base_estimator = base_estimator
        self.score_function = score_function
        self.scale_features = scale_features
        self.estimator = None
        self.selected_features = None
        self.k = None
        self.feature_costs = None

    def fit(self, X, y):
        self.k = int((self.scale_features) * X.shape[1])
        KBest = SelectKBest(self.score_function, self.k)
        KBest = KBest.fit(X, y)
        self.selected_features = KBest.get_support()
        self.estimator = self.base_estimator.fit(X[:, self.selected_features], y)
        return self

    def predict(self, X):
        return self.estimator.predict(X[:, self.selected_features])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[:, self.selected_features])
