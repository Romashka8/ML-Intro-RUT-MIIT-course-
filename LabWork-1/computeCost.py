import numpy as np
from functools import reduce


class MSELoss:

    def __init__(self, weights, x, y):
        """
        weights: array(float), weights = [w_0, w_1]
        model: callable, in this case: f(x) = w_0 + w_1 * x
        """
        self.weights = weights
        self.model = lambda x: self.weights[0] + self.weights[1] * x
        assert len(x) == len(y)
        self.y = y
        self.x = x
        self.dim = (len(y), 1)

    def get_loss(self, method):
        """
        method: one of ('simple', 'func', 'vectorized(returned by default)')
        """
        if method == 'simple':
            return reduce(lambda a, b: a + b, [(self.model(self.x[i]) - self.y[i])**2 for i in range(len(self.x))]) / self.dim[0]
        elif method == 'func':
            return sum([(self.model(self.x[i]) - self.y[i])**2 for i in range(len(self.x))]) / self.dim[0]
        
        return ((np.vectorize(self.model)(np.array(self.x)) - np.array(self.y)) ** 2).mean()
    
    def update_weights(self, weights):
        self.weights = [w for w in weights]


if __name__ == '__main__':
    x = [1, 2, 3]
    y = [4, 5, 6]
    w = [0, 1]

    test = MSELoss(w, x, y)
    print(test.method)
