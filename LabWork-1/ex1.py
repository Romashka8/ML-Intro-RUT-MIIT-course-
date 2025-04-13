import os
import pickle

import numpy as np

from computeCost import MSELoss
from gradientDescent import Optimizer
from plotData import PlotData
from predict import Predictor

from warnings import filterwarnings
filterwarnings('ignore')


def read_data(filename):

    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        data = [list(map(float, row.strip().split(','))) for row in f.readlines()]
    
    return data

def save_weights(filename, weights):
    
    with open(os.path.join(os.getcwd(), filename), 'wb') as f:
        pickle.dump(weights, f)

def train(x, y, iterations=100, learning_rate=0.01, verbose=False, method='simple'):

    opt = Optimizer(x, y, learning_rate)
    loss = MSELoss([1, 1], x, y)

    for iter in range(1, iterations + 1):
        
        weights = opt.step(method)
        loss.update_weights(weights)

        if verbose:
            print(f'Loss на {iter} итерации: ', loss.get_loss(method))
    
    return weights, loss.get_loss(method)

def find_best_model(x, y, grid=None, method='simple'):
    
    grid = grid if grid else {'iterations': np.arange(1, 250, 10), 'leanrning_rate': [0.001, 0.003, 0.01, 0.5, 0.9]}
    best_loss = float('inf')
    best_pair = [None, None]
    bets_weights = [None, None]

    p_1, p_2 = grid.keys()
    
    for iter in grid[p_1]:
        for lr in grid[p_2]:
            weights, loss = train(x, y, iterations=iter, learning_rate=lr, method=method)
            if loss < best_loss:
                best_loss, best_pair, bets_weights = loss, (iter, lr), weights
    
    return bets_weights, best_loss, best_pair


def main(plot=False):

    data = np.array(read_data('ex1data1.txt'))
    x, y = data[:, 0], data[:, 1]
    weights = [1, 1]

    if plot:
        plotter = PlotData(x, y)
        plotter.plot_data()

    method = 'simple' # 'simple', 'func', 'vectorized'
    loss = MSELoss(weights, x, y)

    if plot: print('Стартовый loss: ', loss.get_loss(method))
    
    best_weights, best_loss, best_pair = find_best_model(x, y, method=method)

    if plot: print('Лучший loss: ', best_loss)
    if plot: plotter.plot_data(best_weights)

    save_weights('weights.pkl', best_weights)


if __name__ == '__main__':
    main(plot=True)
