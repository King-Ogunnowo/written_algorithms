import numpy as np

class LogisticRegression():
    """
    Logistic Regression algorithm to predict binary values (y) for data point X
    
    PARAMETERS
    
        - epoch: Number of iterations during training, useful in finding the right  model parameters.
        - learning_rate: degree of wideness of steps to be taken to minimize model error.
        - threshold: decision value to predict 1 or 0. If probability above threshold, then 1, else 0.
        
    HOW THE ALGORITHM PREDICTS:
        1. Initializes 1s for coefficient (weight) and slope (bias).
        Number of weights is determined by number of independent features identified in the data
        2. Compute predictions with the initialized weight, identifies the error in those predictions,\
        and use gradient descent to compute optimal weights. This is done iteratively,\
        and number of iterations is determined by the number of epochs
        3. Optimal weights generated from process in step 2 are used to compute predictions for unseen data
        4. Depending on threshold, 1 or 0 is predicted for instances in the unseen data
        
    NOTEWORTHY FORMULAS:
        1. sigmoid function = 1 / (1 + e(-z))
            where:
                e is exponential function, usually of value 2.718
                z is the result of the logistic equation. Formula us given below
                
        2. linear equation = ßo + ß1 * X1 ... + ßn * Xn
            where: 
                ßo is bias
                ß1 or ßn is weight
                Xn is feature
    """
    def __init__(self, epoch, learning_rate, threshold):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.threshold = threshold
        
    def _initialize_params(self, n_features):
        weight = np.ones(
            shape = n_features, 
            dtype = float
        )
        bias = 1
        return weight, bias
    
    def _sigmoid_function(self, z):
        result = 1 / (
            1 + np.exp(-z)
        )
        return result
    
    def _linear_equation(self, x, weight, bias):
        logistic = self._sigmoid_function(
        np.dot(
            x, weight
        )
            + bias)
        return logistic
    
    def _cost_function(self, y, y_hat):
        cost =-np.mean(
            y * np.log(y_hat) +  (
                    (1 - y) * np.log(1 - y_hat
                                    )
                )
            )
        return cost
    
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
    
    def _accuracy(self, y_hat, y):
        accuracy = np.mean(y == y_hat)
        return accuracy
    
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
        accuracy = self._accuracy(y, prediction)
        loss = self._cost_function(y, y_hat)
        print(f"==== number of epochs: {e}; loss: {loss}; accuracy ==== {accuracy}")
    
    def predict(self, X):
        prediction = [
            1 if value >= 0.5 else 0 for value in self._linear_equation(
                X, self.weight, self.bias
            )
        ]
        return prediction
    
    def predict_proba(self, X):
        probability = self._linear_equation(X, self.weight, self.bias)
        return probability