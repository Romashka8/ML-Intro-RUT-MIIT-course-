import pickle
import os

from predict import Predictor


def load_model(filename):
    weights = pickle.load(open(os.path.join(os.getcwd(), filename), 'rb'))
    return weights


if __name__ == '__main__':

    weights = load_model('weights.pkl')
    pred = Predictor(weights=weights)

    while True:
        n = int(input('Введите число пунктов: '))
        print('Ожидаемая прибыль: ', pred.predict(n))
        out = int(input('Выйти?(0, 1) - '))
        if out == 1: break
