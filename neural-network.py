# neural network is trained, prediction is interpreted manually (not included in code) 
# included print lines for testing and debug. un-comment those lines to see result as you wished

import numpy as np
from numpy.core.numeric import zeros_like

def sigmoid(i): return 1.0 / (1.0 + np.exp(-i))

class Classifier:

    def __init__(self, layers: list, init_val_gen='random') -> None:
        self.layers = layers
        # list of weights
        self.weights = [None] * (len(layers) - 1)
        # list of biases
        self.theta = [None] * (len(layers) - 1)
        gen_func = self._get_gen_func(init_val_gen)
        for i in range(1, len(layers)):
            self.weights[i - 1] = gen_func((layers[i-1], layers[i]))
            self.theta[i - 1] = gen_func((1, layers[i]))

    def _get_gen_func(self, init_val_gen='random'):
        if init_val_gen == 'random':
            return lambda x: np.random.uniform(low=-1, high=1, size=x)
        elif init_val_gen == 'zero':
            return lambda x: np.zeros(size=x)

    def back_propagate(self, c, newWeights, alpha=0.1):
        o = newWeights[-1]
        error = [np.zeros_like(t) for t in self.theta]
        error[-1] = o * (1.0 - o) * (c - o)
        for i in range(len(self.weights) - 2, -1, -1):
            error[i] = newWeights[i] * (1.0 - newWeights[i]) * (error[i + 1]@self.weights[i+1].T)
        for i in range(len(self.weights) - 1, -1, -1):
            self.theta[i] += alpha * error[i]
            weights[i] += alpha * newWeights[i - 1].T * error[i]

    def update_weight(self, x):
        w = x.reshape((1, -1))
        newWeight = [np.zeros_like(t) for t in self.theta]
        for i in range(len(self.weights)):
            # print(w, self.weights[i])
            w = sigmoid(np.dot(w, self.weights[i]) + self.theta[i])
            newWeight[i] = w
        return newWeight

    def train(self, X: np.matrix, y, epoch=1000, alpha=0.1):
        for _ in range(epoch):
            for idx in range(X.shape[0]):
                newWeights = self.update_weight(X[idx, :].reshape((1, -1)))
                # print(newWeights)
                self.back_propagate(y[idx, :], newWeights, alpha=alpha)
                # print(self.weights)

if __name__ == "__main__":
    # Test case 
    model = Classifier([3, 2, 1]) # input layer has 3 nodes, hidden layer has 2 nodes, output layer has 1 node

    # weights and biases of nodes are stored in matrices as described below
    weights = [
        np.array([
            [0.2, -0.1], # w14 and w15
            [0.3, 0.2],  # w24 and w25
            [-0.4, 0.3]  # w34 and w35
        ]),
        np.array([
            [0.2], # w46
            [-0.1]  # w56
        ])
    ]
    bias = [
        np.array(
            [[-0.1, 0.3]] # theta4 and theta5
        ),
        np.array(
            [[0.2]] # theta6
        )
    ]
    model.weights = weights
    model.bias = bias
    model.train(np.array([[1.0, 0.0, 1.0]]), np.array([[1]]), epoch=1,alpha=0.9)
    print(model.weights)
    print(model.bias)