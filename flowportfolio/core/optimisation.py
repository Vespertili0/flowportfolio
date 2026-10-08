from typing import Any

from skfolio.optimization import BaseOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class SkfolioGridSearchCV(BaseOptimization, GridSearchCV):
    """A wrapper for GridSearchCV that conforms to skfolio's BaseOptimization API.

    This ensures that when skfolio's cross_val_predict uses this search object,
    it exposes weights_ and predict correctly as an optimization estimator.
    """

    def __init__(self, estimator: Any = None, param_grid: Any = None, **kwargs: Any):
        if estimator is None:
            # When cloned by scikit-learn without parameters? Shouldn't happen unless estimator/param_grid aren't in get_params()
            pass
        GridSearchCV.__init__(
            self, estimator=estimator, param_grid=param_grid, **kwargs
        )
        BaseOptimization.__init__(self)

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "SkfolioGridSearchCV":
        GridSearchCV.fit(self, X, y, **fit_params)

        # In scikit-learn Pipelines, the weights_ attribute resides in the final estimator
        if isinstance(self.best_estimator_, Pipeline):
            self.weights_ = self.best_estimator_.steps[-1][1].weights_
            if hasattr(self.best_estimator_.steps[-1][1], "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.steps[-1][
                    1
                ].feature_names_in_
        else:
            self.weights_ = self.best_estimator_.weights_
            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        return self
