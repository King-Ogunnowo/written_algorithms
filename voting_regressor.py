import scipy.stats as sc
    

class Voting_Regressor:
    
    """
    Voting Regressor:
    
    Makes use of the bagging ensemble method (different algorithms on the same data) to predict continuous values
    It does this by training the algorithms on the data individually,
    and aggregating their predictions into a single prediction for each row. 
    
    It computes a prediction by averaging the continous predictions of each algorithm.
                       
        PARAMETERS
        ----------
            estimators: list of regressors
            
        RETURNS
        -------
            array of predictions
    
    """
    def __init__(self, estimators):
        self.estimators = estimators
        
    def fit(self, X, y):
        self.fitted_estimators = [
            estimator.fit(
                X, y
            ) for estimator in self.estimators
        ]
        return self
    
    def predict(self, X):
        
        y_hats = np.empty(
                (
                    len(
                        self.fitted_estimators
                    ), 
                    len(X)
                )
            )
        for i, estimator in enumerate(self.fitted_estimators):
            y_hats[i] = estimator.predict(X)
        return np.mean(y_hats, axis = 0)
            