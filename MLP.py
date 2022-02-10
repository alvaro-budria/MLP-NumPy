import numpy as np
import pickle
from typing import List


def crossEntropyLoss(y: np.array, p: np.array) -> float:
    return - (y * np.log(p + 1e-9)).sum(axis=1).sum(axis=0)


def sigmoid(z: np.array) -> np.array:
    return 1/(1 + np.exp(-z))


def derivative_sigmoid(z: np.array) -> np.array:
    return sigmoid(z) * (1 - sigmoid(z))


def ReLU(z: np.array) -> np.array:
    return (abs(z) + z) / 2


def derivative_ReLU(z: np.array) -> np.array:
    return np.greater(z, 0).astype(int)


def LReLU(z: np.array, alpha: float = 0.01) -> np.array:
    return np.where(z > 0, z, z * alpha)


def derivative_LReLU(z: np.array, alpha: float = 0.01) -> np.array:
    return np.greater(z, 0).astype(int) + alpha * np.less_equal(z, 0).astype(int)


class MLP:

    def __init__(
        self,
        layer_size: List[int],
        input_size: int,
        hidAct: str = 'ReLU',
        lastLayerAct: str = 'identity',
        dropout: float = 0
    ):

        self.hidAct = hidAct
        self.lastLayerAct = lastLayerAct
        self.activation = {
            'ReLU': ReLU,
            'LReLU': LReLU,
            'sigmoid': sigmoid,
            'identity': lambda x: x,
        }
        self.derivative = {
            'ReLU': derivative_ReLU,
            'LReLU': derivative_LReLU,
            'sigmoid': derivative_sigmoid,
            'identity': lambda _: 1,
        }

        assert(0 <= dropout < 1)
        self.dropout = dropout

        self.update_params = True

        self.layer_size = layer_size
        self.input_size = input_size

        self.layer_size.insert(0, input_size)
        # first layer of neurons is an input layer
        # use He's wieght initialization: sqrt(2/size_previous_layer) 
        self.W = [np.random.normal(0, np.sqrt(2/layer_size[i]), size=(s, ss))
                  for i, (s, ss) in enumerate(zip(layer_size[1:], layer_size[:-1]))]
        self.B = [0.01*np.ones(s) for s in layer_size[1:]]
        self.Z = [None for s in layer_size]
        self.A = [None for s in layer_size]
        self.D = [None for s in layer_size]

    def D_outLayer(
        self,
        output_activations: np.array,
        y: np.array
    ) -> np.array:
        return (output_activations - y)

    def feedforward(
        self,
        input: np.array
    ) -> np.array:
        self.A[0] = input.reshape(-1, self.input_size)

        for i, (w, b) in enumerate(zip(self.W[:-1], self.B[:-1])):
            self.Z[i+1] = np.einsum('ij,kj->k...i', w, self.A[i]) + b
            # self.A[i+1] = self.activation[self.hidAct](self.Z[i+1])
            self.A[i+1] = self.dropout_layer(self.activation[self.hidAct](self.Z[i+1]))

        self.Z[-1] = np.einsum('ij,kj->k...i', self.W[-1], self.A[-2]) + self.B[-1]
        self.A[-1] = self.activation[self.lastLayerAct](self.Z[-1])
        return self.A[-1]

    def dropout_layer(self, a: np.array) -> None:
        if not self.update_params:
            return a
        binary_mask = np.random.rand(*a.shape) < 1 - self.dropout
        a = np.multiply(a, binary_mask)
        return a / (1 - self.dropout)

    def backprop(self, y: np.array) -> None:
        self.D[-1] = self.D_outLayer(self.A[-1], y.reshape(-1, self.layer_size[-1])) * self.derivative[self.lastLayerAct](self.Z[-1])
        for l in range(2, len(self.layer_size)):
            delta = np.einsum('ji,kj->k...i', self.W[-l+1], self.D[-l+1]) * self.derivative[self.hidAct](self.Z[-l])
            self.D[-l] = delta

    def backward(
        self,
        y: np.array,
        lr: float = 0.01,
        wd: float = 0.01
    ) -> None:
        if not self.update_params:
            raise ValueError('Cannot update model\'s parameters when the model is in inference mode. To train your model, use .test() method.')

        self.backprop(y)
        for l in range(1, len(self.layer_size)):
            assert(self.A[-l-1].shape[0] == self.D[-l].shape[0])
            self.W[-l] = (1 - lr * wd) * self.W[-l] - (lr / self.D[-l].shape[0]) * np.einsum('...i,...j->...ij', self.D[-l], self.A[-l-1]).sum(0)
            self.B[-l] -= (lr / self.D[-l].shape[0]) * self.D[-l].sum(0)

    def train(self) -> None:
        self.update_params = True

    def test(self) -> None:
        self.update_params = False

    def save(self, filepath: str) -> None:
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during model saving:", ex)

    @staticmethod
    def load(filepath: str) -> 'MLP':
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as ex:
            print("Error during model loading:", ex)

    def __call__(self, input: np.array) -> 'MLP':
        return self.feedforward(input)


if __name__ == '__main__':
    from utils import batch_generator
    np.random.seed(41)
    my_MLP = MLP(layer_size=[32, 16, 1], input_size=10, hidAct='LReLU', dropout=0)
    my_MLP.train()

    # sample dummy data to test if MLP can overfit train data
    X = np.array([[-100]*10, [100]*10, [-100]*10, [0]*10])
    Y = np.array([-50, 100, -50, 0])

    for epoch in range(4000):
        print(f'----- epoch {epoch} -----')
        for x, y in batch_generator(X, Y, 4, shuffle=False):
            pred = my_MLP(x)
            print(pred)
            my_MLP.backward(y, lr=0.000005, wd=0.1)

    print('---- on test mode ----')
    my_MLP.test()
    pred = my_MLP(x)
    print(pred)