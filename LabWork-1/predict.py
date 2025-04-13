import numpy as np


class Predictor:

    def __init__(self, weights=[1, 1]):
        
        self.weights = weights

    def predict(self, x, weights=None):

        self.weights = weights if weights else self.weights

        return self.weights[0] + self.weights[1] * x
