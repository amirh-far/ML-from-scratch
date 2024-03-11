import numpy as np
import math

class BatchGDLogisticRegression():
    def __init__(self, learning_rate, n_epoch) -> None:
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
    

    def initialize_weights(self, n_features) -> None:
        """ initialize the weights randomly (from eriklindernoren/ml-from-scratch)"""
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (1, n_features))
    

    # sigmoid function
    def sigmoid(self, x:np.ndarray):
        return 1 / (1 + np.exp(-x))


    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        # insert a dummy value to the X matrix for the intercept
        X_b = np.insert(arr=X, obj=0, values=1, axis=1)

        n_features = X_b.shape[1]
        # initialize the weights with random values.
        self.initialize_weights(n_features=n_features)

        for epoch in range(self.n_epoch):
            gradient = (1 / n_features) * X_b.T @ (self.sigmoid(X_b @ self.w) - y)

            self.w = self.w - gradient * self.learning_rate


    def predict(self, X:np.ndarray) -> np.ndarray:
        X_b = np.insert(arr=X, obj=0, values=1, axis=1)

        # predict the values via the weights, input matrix using sigmoid function
        # and then, classify the predictions
        y = np.round(self.sigmoid(X_b @ self.w)).astype(int)
        
        # y = np.array([1 if pred >= 0.5 else 0 for pred in pred_values])
        return y



        




        
