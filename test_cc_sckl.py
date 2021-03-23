import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston

'''class MeanRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = y.mean()
        return self
    def predict(self, X):
        return np.array(X.shape[0] * [self.mean_])


X, y = load_boston(return_X_y=True)
l = MeanRegressor()
l.fit(X, y)
print(l.predict(X))'''


class NullRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        # The prediction will always just be the mean of y
        self.y_bar_ = np.mean(y)
    def predict(self, X=None):
        # Give back the mean of y, in the same
        # length as the number of X observations
        return np.ones(X.shape[0]) * self.y_bar_

class NullClassifier(ClassifierMixin):
    def __int__(self, c=None):
        self.c = c
    def fit(self,X=None, y=None):
        if (np.mean(X+y)%2) == 0:
            self.y_bar_ = True
        else:
            self.y_bar_ = False
        return self.y_bar_
    def predict(self, X=None):
        return np.full(X.shape,self.y_bar_)


class ConstantRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, c=None):
        self.c = c

    def fit(self, X, y):
        if self.c is None:
            self.const_ = y.mean()
        else:
            self.const_ = self.c
        return self

    def predict(self, X):
        return np.array(X.shape[0] * [self.const_])

'''X_train = np.array([1,1,2])
X_test = np.array([2,2,2,2,2,2])
Y_train = np.array([1,1,1])
#model = NullRegressor()
model = NullClassifier()
print(model.fit(X_train,Y_train))
print(model.predict(X_test))'''

X, y = load_boston(return_X_y=True)
'''l = ConstantRegressor(10.)
l.fit(X, y)
l.predict(X)'''

parameters = {'c': np.linspace(0, 50, 100)}
grid = GridSearchCV(ConstantRegressor(),parameters)
grid.fit(X,y)
print(grid.best_params_)

#(estimator=ConstantRegressor(),param_grid={'c': np.linspace(0, 50, 100)},)grid.fit(X, y)