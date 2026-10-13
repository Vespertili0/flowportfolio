from sklearn.base import BaseEstimator, clone

class MyEstimator(BaseEstimator):
    def __init__(self, a=1, **kwargs):
        self.a = a
        for k, v in kwargs.items():
            setattr(self, k, v)

est = MyEstimator(a=2, b=3)
print("Params:", est.get_params())
