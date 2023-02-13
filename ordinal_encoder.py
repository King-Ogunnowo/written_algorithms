import warnings
import numpy as np
import pandas as pd


class OrdinalEncoder:

    """
    Ordinal Encoder class object to encode strings with integers
    Makes use of a nested dictionary to accomplish this

    Methods of this class include:
        fit(): For the transformer to know the string values in the data
        transform(): To transform the string values to integers
        fit_transform(): To combine fit and transform functionalities together
        inverse_transform(): To revert the encoding process, transform numbers back to the original string
        partial_fit():To apply the transformer on new data, without loosing previous information
    """

    def __init__(self):
        pass

    def fit(self, X, y = None):
        self.features = X.columns.tolist()
        self.dictionary = {}
        for feature in self.features:
            unique_values = X[feature].unique()
            self.dictionary[feature] = {
                key:value for key, value in zip(
                    unique_values, 
                    range(len(unique_values))
                    )
            }
        print(f"transformer object has been fitted")

    def transform(self, X):
        X_transformed = X.copy()
        if X.columns.tolist() == self.features:
            for feature in self.features:
                X_transformed[feature] = X_transformed[feature].map(self.dictionary[feature])
            return X_transformed
        else:
            warnings.warn('Features in data paassed different from features expecting')
            print(
                f'features passed: {X.columns.tolist()}, expected features: {self.features}, cannot proceed!'
                )

    def fit_transform(self, X, y = None):
        self.features = X.columns.tolist()
        self.dictionary = {}
        for feature in self.features:
            unique_values = X[feature].unique()
            self.dictionary[feature] = {
                key:value for key, value in zip(
                    unique_values,
                     range(len(unique_values))
                     )
            }
        print(f"transformer object has been fitted")
        X_transformed = X.copy()
        if X.columns.tolist() == self.features:
            for feature in self.features:
                X_transformed[feature] = X_transformed[feature].map(
                    self.dictionary[feature]
                    )
            return X_transformed
        else:
            warnings.warn('Features in data paassed different from features expecting')
            print(f'features passed: {X.columns.tolist()}, expected features: {self.features}, cannot proceed!')

    def inverse_transform(self, X):
        X_reversed = X.copy()
        for feature in self.features:
            feature_dictionary_object = {
                value:key for key, value in self.dictionary[feature].items()
            }
            print(feature_dictionary_object)
            X_reversed[feature] = X_reversed[feature].map(feature_dictionary_object)
        return X_reversed

    def partial_fit(self, X):
        for feature in self.features:
            unique_values = X[feature].unique().tolist()
            max_value = max(self.dictionary[feature].values()) + 1
            initial_dictionary = {
                key:max_value + value for key, value in zip(unique_values, range(len(unique_values)))
            }
            self.dictionary[feature].update(initial_dictionary)
        print(f"transformer object has been fitted")
