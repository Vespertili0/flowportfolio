from sklearn.model_selection import GridSearchCV
from skfolio.optimization import BaseOptimization
from sklearn.pipeline import Pipeline

class SkfolioGridSearchCV(BaseOptimization, GridSearchCV):
    def __init__(self, estimator=None, param_grid=None, **kwargs):
        if estimator is None:
            pass
        GridSearchCV.__init__(
            self, estimator=estimator, param_grid=param_grid, **kwargs
        )
        BaseOptimization.__init__(self)

print("MRO:", [c.__name__ for c in SkfolioGridSearchCV.mro()])
search = SkfolioGridSearchCV(estimator=Pipeline([('a', None)]), param_grid={})
print("Success")
