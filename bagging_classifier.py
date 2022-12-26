import numpy as np
import scipy.stats as sc
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class BaggingClassifier:
    
    """
    Bagging classifier class object
    To create an ensemble model of n_estimators.
    
    PARAMETERS
    ----------
        estimator: Base estimator for usage. Typical bagging methods use decision trees.
                   However, you are free to experiment with other base estimators. E.g. Logistic Regression
                   
        n_estimators: Number of etimator instances to train in the bagging method. Usually this is 100
        
        replacement: Controls the frequency by which an instance of data can be sampled.
                     If set to true, a data point can be chosen as part of a sample more than once
                     
    RETURNS
    -------
        returns a trained Bagging Classifier
    """
    
    def __init__(self, estimator = DecisionTreeClassifier(random_state = 42), n_estimators = 100, replacement = True):
        
        self.estimator = estimator
        self.N, self.D = X.shape
        self.n_estimators = n_estimators
        self.replacement = replacement
        self.fitted_algs = []
        
    def _mode(self, predictions):
        
        return Counter(
            predictions
        ).most_common(1)[0][0]
        
    def fit(self, X, y):
        for n in range(self.n_estimators):
            
            sample = np.random.choice(
                np.arange(
                    self.N
                ), 
                size = self.N, 
                replace = self.replacement
            )
        
            self.fitted_algs.append(
                self.estimator.fit(X[sample], y[sample])
            )
        return self
    
    def predict(self, X):
        
        y_hats = np.empty(
            (
                len(
                    self.fitted_algs
                ), 
                len(X)
            )
        )
        
        for i, estimator in enumerate(self.fitted_algs):
            y_hats[i] = estimator.predict(X)
        return sc.mode(y_hats)[0].ravel()