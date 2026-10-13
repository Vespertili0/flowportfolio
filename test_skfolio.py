from flowportfolio.core.optimisation import SkfolioGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

search = SkfolioGridSearchCV(estimator=Pipeline([('a', None)]), param_grid={}, cv=5)
print("Params:", search.get_params())
try:
    cloned = clone(search)
    print("Cloned cv:", getattr(cloned, 'cv', 'missing'))
except Exception as e:
    print("Error:", type(e), e)
