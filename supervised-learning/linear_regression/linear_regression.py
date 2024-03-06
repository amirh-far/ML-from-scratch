import numpy as np
import math

class GradientDescent:
    pass

# simplest form of linear regression via normal equation and least squares as cost function
class LinearRegression:
    def __init__(self) -> None:
        self.w = None

    def fit(self, X, y) -> None:
        # adding intercept term(dummy value) for bias
        X_b = np.insert(arr=X, obj=0, values=1, axis=1)
        
        # calculating the parameter vector, minimizing the mean squared error via Normal Equation:
        self.w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.insert(arr=X, obj=0, values=1, axis=1)

        y_pred = X_b @ self.w
        return y_pred

# linear regression applied with batch gradient descent
class BatchGradientLinearRegression:
    def __init__(self, learning_rate=0.01, n_epoch=100) -> None:
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
    
    def initialize_weights(self, n_features) -> None:
        """ initialize the weights randomly (from eriklindernoren)"""
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))        
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X_b = np.insert(arr=X, obj=0, values=1, axis=1)
        n_features = X.shape[1]
        # initialize the weight with random numbers
        self.initialize_weights(n_features=n_features)

        # train ALL of data on each iteration so it can be called epoch
        for epoch in range(self.n_epoch):
            # calculating the gradient vector
            gradient = (2 / n_features) * X_b.T @ (X_b @ self.w - y)

            self.w = self.w - gradient * self.learning_rate

    def predict(self, X):
        X_b = np.insert(arr=X, obj=0, values=1, axis=1)

        y_pred = X_b @ self.w
        return y_pred

            










