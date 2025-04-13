import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class PlotData:

    def __init__(self, x, y, weights=[0, 0]):
        """
        weights: array(float), weights = [w_0, w_1]
        model: callable, in this case: f(x) = w_0 + w_1 * x
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.weights = weights
        self.model = np.vectorize(lambda x: self.weights[0] + self.weights[1] * x)

    def plot_data(self, weights=None):
        
        self.weights = weights if weights else self.weights
        model = self.model(self.x)

        if sum(self.weights) != 0:
            plt.plot(self.x, model, color='indigo', label='f(x)')
        
        plt.scatter(self.x, self.y, color='deeppink', label='data')
        plt.scatter([], [], label=f'weights: w_0 = {self.weights[0]:.2f}; w_1 = {self.weights[1]:.2f}', color='lavender')

        plt.title('Начальное распределение и линия регрессии')

        plt.legend()
        plt.grid(axis='y')
        plt.show()


if __name__ == '__main__':

    x = [1, 2, 3]
    y = [4, 5, 6]

    test = PlotData(x, y)
    test.plot_data(weights=[0, 1])
