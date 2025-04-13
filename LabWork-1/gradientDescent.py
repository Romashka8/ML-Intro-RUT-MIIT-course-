import numpy as np
from functools import reduce


class Optimizer:
    
    def __init__(self, x, y, learning_rate=0.01):
        """
        weights: array(float), weights = [w_0, w_1]
        model: callable, in this case: f(x) = w_0 + w_1 * x
        """
        self.weights = [1, 1]
        self.model = lambda x: self.weights[0] + self.weights[1] * x
        self.learning_rate = learning_rate
        assert len(x) == len(y)
        self.y = y
        self.x = x
        self.dim = (len(y), 1)

    def step(self, method='vectorized'):
        """
        method: one of ('simple', 'func', 'vectorized(returned by default)')
        """
        if method == 'simple':
                grad =  [
                        2 * reduce(lambda a, b: a + b, [(self.model(self.x[i]) - self.y[i]) for i in range(len(self.x))]) / self.dim[0],
                        2 * reduce(lambda a, b: a + b, [(self.model(self.x[i]) - self.y[i]) * self.x[i] for i in range(len(self.x))]) / self.dim[0]
                    ]
        elif method == 'func':
            grad = [
                        2 * sum([(self.model(self.x[i]) - self.y[i]) for i in range(len(self.x))]) / self.dim[0],
                        2 * sum([(self.model(self.x[i]) - self.y[i]) * self.x[i] for i in range(len(self.x))]) / self.dim[0]
                    ]
        else:
            # col with ones for intercept
            x_vec = np.column_stack([np.ones(self.dim[0]), self.x])
            error = x_vec @ np.array(self.weights) - np.array(self.y)
            grad = 2 * x_vec.T @ error / self.dim[0]

        self.weights = [self.weights[i] - self.learning_rate * grad[i] for i in range(len(grad))]

        return self.weights
    

if __name__ == '__main__':

    x = [1, 2, 3]
    y = [4, 5, 6]

    test = Optimizer(x, y)
    print(test.step('simple'))
    print(test.step('func'))
    print(test.step())
