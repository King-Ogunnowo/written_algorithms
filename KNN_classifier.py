import math
import numpy as np
from collections import Counter

class KNNClassifier():
    """
    K Nearest Neighbours Classifier:
    
    Predicts a discrete value (binary or multiple)
    for a data instance based on how frequently 
    the value occurs in the closest K training instances
    
    it computes the closes K by using either euclidean or manhattan distance metrics
    
    NOTE WORTHY FORMULAS:
    
    1. Euclidean: Square root of the sum of squared differences between two points. 
                  In this case, two data instances
                  
                  √(q - p) ** 2
                  
                  in case of multiple points
                  
                  √((q_1 - p_1) ** 2) + ... + ((q_n - p_n) ** 2)
                  
    2. Manhattan: Sum of the absolute differences between two data instances
                  
                  |p - q|
                  
                  in case of multiple intances
                  
                  |p_1 - q_1| + ... + ... |p_n - q_n|
                  
    PARAMETERS
    ----------
    
    1. K = Number of closest instances to consider for predict() or predict_proba()
    
    2. distance_func = distance metric to identify closest and farthest instances. 
                       either euclidean or manhattan
    """
    def __init__(self, K, distance_func):
        self.K = K
        self.distance_func = distance_func
        
    def _euclidean_distance(self, new_point, old_point):
        new_point = np.array(new_point)
        old_point = np.array(old_point)
        distance = math.sqrt(
            sum(
                (old_point - new_point)
                ** 2)
        )
        return distance
    
    def _manhattan_distance(self, new_point, old_point):
        new_point = np.array(new_point)
        old_point = np.array(old_point)
        distance = sum(
            abs(
                old_point - new_point
            )
        )
        return distance
    
    def _compute_neighbours(self, X, y, new_instance):
        if self.distance_func == 'euclidean':
            computed_distance = [
                (
                    self._euclidean_distance(
                        X.iloc[row], new_instance
                    ), 
                    target
                ) for row, target in zip(range(len(X)), y)]
            return sorted(computed_distance)[:self.K]
        if self.distance_func == 'manhattan':
            computed_distance = [
                (
                    self._manhattan_distance(
                        X.iloc[row], new_instance
                    ), 
                    target
                ) for row, target in zip(range(len(X)), y)]
            return sorted(computed_distance)[:self.K]
        
    def _most_freq_class(self, class_list):
        classes = [
            class_ for distance, class_ in class_list
        ]
        return Counter(classes).most_common(1)[0][0]
                
    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        self.X = X
        self.y = y
        return self
    
    def predict(self, X):
        prediction = []
        for row in range(len(X)):
            prediction.append(self._most_freq_class(
                    self._compute_neighbours(
                        self.X, self.y, X.iloc[row]
                    )
                ))
        return prediction
    
    def predict_proba(self, X):
        probability = []
        for row in range(len(X)):
            computed_distances = self._compute_neighbours(
                        self.X, self.y, X.iloc[row]
                    )
            classes = pd.Series(
                [
                    class_ for distance, class_ in computed_distances
                ]
            )
            if len(self.unique_classes) == 2:
                probability.append(
                    np.mean(
                        classes == 1
                    )
                )
            if len(self.unique_classes) > 2:
                probability.append(
                    max([np.mean(classes == class_ for class_ in self.unique_classes)])
                )
        return probability
