import numpy as np
import math

class LinearRegression:
    """
    Linear Regression algorithm to predict continuous values (y) for data point X
    
    PARAMETERS
    
        - epoch: Number of iterations during training, useful in finding the right  model parameters.
        - learning_rate: degree of wideness of steps to be taken to minimize model error.
        
    HOW THE ALGORITHM PREDICTS:
        1. Initializes 1s for coefficient (weight) and slope (bias).
        Number of weights is determined by number of independent features identified in the data
        2. Compute predictions with the initialized weight, identifies the error in those predictions,\
        and use gradient descent to compute optimal weights. This is done iteratively,\
        and number of iterations is determined by the number of epochs
        3. Optimal weights generated from process in step 2 are used to compute predictions for unseen data
        
    DEFAULT HYPERPARAMETER CONFIGURATIONS
    The default hyperparameter configurations for this algorithm are as follows:
        1. epoch (Number of iterations): 1000
        2. learning_rate (Controls the gradient descent mechanism used to update model weights): 0.03

    NOTEWORTHY FORMULAS:
                
        1. linear equation = ßo + ß1 * X1 ... + ßn * Xn
            where: 
                ßo is bias
                ß1 or ßn is weight
                Xn is feature
    """
    def __init__(self, epoch = 1000, learning_rate = 0.03):
        self.epoch = epoch
        self.learning_rate = learning_rate
        
    def _get_params(self):
        params_dict = {
            'epoch':self.epoch,
            'learning_rate':self.learning_rate
        }
        return params_dict
    
    def _initialize_params(self, n_features):
        weight = np.ones(
            shape = n_features, 
            dtype = float
        )
        bias = 1
        return weight, bias
    
    def _linear_equation(self, x, weight, bias):
        result = np.dot(
            x, weight
        ) + bias
        return result
    
    def _cost_function(self, y, y_hat):
        mse = sum(
            (
                y - y_hat
            ) ** 2
        )
        rmse = math.sqrt(mse)
        return mse
    
    def _gradient_descent(self, n_samples, y, y_hat, X, learning_rate):
        
        dW = ((
            1/n_samples
        ) * np.dot(
            X.T, (y_hat - y)
        )) * learning_rate
        dB = (
            1/n_samples
        ) * np.sum(
            (y_hat - y)
        ) * learning_rate
        return dW, dB
    
    def fit(self, X, y):
        y = np.array(y)
        n_samples, n_features = X.shape[0], X.shape[1]
        self.weight, self.bias = self._initialize_params(n_features)
        for e in range(self.epoch):
            y_hat = self._linear_equation(
                X, 
                self.weight, 
                self.bias
            )
            weight, bias = self._gradient_descent(
                n_samples, 
                y, 
                y_hat, 
                X, 
                self.learning_rate
            )
            
            self.weight -= weight
            self.bias -= bias
        prediction = self.predict(X)
        loss = self._cost_function(y, y_hat)
        print(f"==== number of epochs: {e}; loss: {np.round(loss, 10)}")
    
    def predict(self, X):
        prediction = self._linear_equation(X, self.weight, self.bias)
        return prediction