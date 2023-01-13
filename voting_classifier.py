import scipy.stats as sc
    

class Voting_Classifier:
    
    """
    Voting Classifier:
    
    Makes use of the bagging ensemble method (different algorithms on the same data)
    It does this by training the algorithms on the data individually,
    and aggregating their predictions into a single prediction for each row. 
    
    The method of aggregation used is called voting. 
    Voting is of two types:
        - Hard Voting: Assigns the most frequent prediction per row. 
                       e.g. if row 1 has predictions [1, 1, 0] for three classifiers, the prediction for row 1 is 1.
                       
        - Soft Voting: Makes use of the average probabilities computed by each model, for each data instance.
                       e.g. if row 1 has probabilities [0.90, 0.80, 0.40], all representing the likelihood of the predicted value being 1,
                       aggregating the probabilities to give a single propability is defined as (0.90 + 0.80 + 0.40)/ 3
                       
        PARAMETERS
        ----------
            estimators: list of classifiers
            voting_strategy: string, either soft or hard
            
        RETURNS
        -------
            array of predictions
    
    """
    def __init__(self, estimators, voting_strategy, weights):
        self.estimators = estimators
        self.voting_strategy = voting_strategy
        
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
        
        if self.voting_strategy == 'hard':
            for i, estimator in enumerate(self.fitted_estimators):
                y_hats[i] = estimator.predict(X)
            return sc.mode(y_hats, keepdims = True)[0].ravel()
        
        if self.voting_strategy == 'soft':
            proba = []
            for i, estimator in enumerate(self.fitted_estimators):
                y_hats[i] = estimator.predict_proba(X)[:, 1]
            return np.mean(y_hats, axis = 0)
            