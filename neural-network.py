# neural network is trained, prediction is interpreted manually (not included in code) 
# included print lines for testing and debug. un-comment those lines to see result as you wished

import numpy as np
from numpy.core.numeric import zeros_like
from numba import njit

def sigmoid(i): return 1.0 / (1.0 + np.exp(-i))

def back_propagate(weights, theta, layers, c, outputs, alpha=0.1):
        error = [np.zeros_like(t) for t in theta]
        o = outputs[-1]
        error[-1] = o * (1.0 - o) * (c - o)
        for i in range(len(weights) - 2, -1, -1):
            error[i] = outputs[i] * (1.0 - outputs[i]) * (error[i + 1]@weights[i+1].T)
        for i in range(len(weights) - 1, -1, -1):
            weights[i] += alpha * outputs[i - 1].T * error[i]
            theta[i] += alpha * error[i]

def feed_forward(weights, theta, x):
        outputs = [np.zeros_like(t) for t in theta]
        output = x.reshape((1, -1))
        for i in range(len(weights)):
            #print(output, weights[i])
            output = sigmoid(np.dot(output, weights[i]) + theta[i])
            outputs[i] = output
        return outputs

def fit(weights, theta, layers, X, y, epoch=1000, alpha=0.1): # choosing 0.1 as the default value
    for _ in range(epoch):
        for idx in range(X.shape[0]):
            #print(X[idx, :].reshape((1, -1)))
            outputs = feed_forward(weights, theta, X[idx, :])
            #print(outputs)
            back_propagate(weights, theta, layers, y[idx, :], outputs, alpha=alpha)
            #print(self.weights)

class Classifier:

    def __init__(self, layers: list, init_val_gen='random') -> None:
        self.layers = layers
        # list of weights
        self.weights = [None] * (len(layers) - 1)
        # list of biases
        self.bias = [None] * (len(layers) - 1)
        gen_func = self._get_gen_func(init_val_gen)
        for i in range(1, len(layers)):
            
            self.weights[i - 1] = gen_func((layers[i-1], layers[i]))
            self.bias[i - 1] = gen_func((1, layers[i]))
            
    def train(self, X: np.matrix, y, epoch=1000, alpha=0.1):
        fit(self.weights, self.bias, self.layers, X, y, epoch, alpha)
            
    def _feed_forward(self, x):
        return feed_forward(self.weights, self.bias, x)

    def _back_propagate(self, y, outputs, alpha=0.1):
        back_propagate(self.weights, self.bias, self.layers, y, outputs, alpha)

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
    # print(model.weights)
    # print(model.bias)