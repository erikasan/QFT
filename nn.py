import numpy as np

n_classes = 2

X = np.random.rand(2, 2)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def feed_forward(X, n_classes):
    W = np.random.rand(n_classes, X.shape[0])
    b = np.random.rand(X.shape[0], 1)

    a = W@X + b

    return sigmoid(a)

class Node:
    def __init__(self, activation_func = None, weights = None, bias = None):
        if activation_func is not None:
            assert callable(activation_func), 'activation_func must be callable'

        activation_func = self.activation_func
        weights = self.weights

    def set_activation_func(self, activation_func):
        self.activation_func = activation_func

    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias

class Layer:
    def __init__(self, nodes = None, n_inputs = None, weight_matrix = None, l = None):
            self.nodes = nodes
            self.n_inputs = n_inputs

            if weight_matrix is not None:
                assert weight_matrix.shape == (nodes, n_inputs)
            self.weight_matrix = weight_matrix

            self.l = l

    def initialize(self):
        assert self.nodes is not None, 'self.nodes must not be None to initialize weights and biases'
        assert self.n_inputs is not None, 'self.n_inputs must not be None to initialize weights and biases'

        self.weight_matrix = np.random.randn(self.nodes, self.n_inputs)
        self.bias = 0.01*np.ones((self.nodes, 1))

class NeuralNetwork:
    def __init__(self, n_inputs, n_outputs, layers = None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layers = layers
        pass
