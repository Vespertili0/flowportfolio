from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge

class SkfolioGridSearchCV(BaseEstimator):
    def __init__(self, estimator=None, param_grid=None, **kwargs):
        GridSearchCV.__init__(self, estimator=estimator, param_grid=param_grid, **kwargs)

search = SkfolioGridSearchCV(estimator=Ridge(), param_grid={'alpha': [1.0]}, cv=5)
try:
    print("Params:", search.get_params())
    cloned = clone(search)
    print("Cloned cv:", getattr(cloned, 'cv', 'missing'))
except Exception as e:
    print("Error:", e)
