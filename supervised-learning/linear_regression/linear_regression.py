import numpy as np


class LinearRegression:
    def __init__(self) -> None:
        self.w = None

    def fit(self, X, y) -> None:
        # adding intercept term(dummy value) for bias
        X_b = np.insert(arr=X, obj=0, values=1, axis=1)
        
        # calculating the parameter vector, minimizing the mean squared error via this function:
        self.w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.insert(arr=X, obj=0, values=1, axis=1)

        y_pred = X_b @ self.w
        return y_pred
        




