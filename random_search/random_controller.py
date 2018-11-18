from keras import models, layers, backend, initializers
import tensorflow as tf
from transformations import get_transformations
import PIL.Image
import numpy as np

LSTM_UNITS = 100

SUBPOLICIES = 5
SUBPOLICY_OPS = 2

OP_TYPES = 16
OP_PROBS = 11
OP_MAGNITUDES = 10

class Operation:
    def __init__(self, X, types_softmax, probs_softmax, magnitudes_softmax, argmax=False):
        # Ekin Dogus says he sampled the softmaxes, and has not used argmax
        # We might still want to use argmax=True for the last predictions, to ensure
        # the best solutions are chosen and make it deterministic.
        transformations = get_transformations(X)
        if argmax:
            self.type = types_softmax.argmax()
            t = transformations[self.type]
            self.prob = probs_softmax.argmax() / (OP_PROBS-1)
            m = magnitudes_softmax.argmax() / (OP_MAGNITUDES-1)
            self.magnitude = m*(t[2]-t[1]) + t[1]
        else:
            self.type = np.random.choice(OP_TYPES, p=types_softmax)
            t = transformations[self.type]
            self.prob = np.random.choice(np.linspace(0, 1, OP_PROBS), p=probs_softmax)
            self.magnitude = np.random.choice(np.linspace(t[1], t[2], OP_MAGNITUDES), p=magnitudes_softmax)
        self.transformation = t[0]

    def __call__(self, X):
        _X = []
        for x in X:
            if np.random.rand() < self.prob:
                x = PIL.Image.fromarray(x)
                x = self.transformation(x, self.magnitude)
            _X.append(np.array(x))
        return np.array(_X)

    def __str__(self):
        return 'Operation %2d (P=%.3f, M=%.3f)' % (self.type, self.prob, self.magnitude)

class Subpolicy:
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, X):
        for op in self.operations:
            X = op(X)
        return X

    def __str__(self):
        ret = ''
        for i, op in enumerate(self.operations):
            ret += str(op)
            if i < len(self.operations)-1:
                ret += '\n'
        return ret

class Controller:
    def __init__(self):
	pass

    def create_model(self):
        # Implementation note: Keras requires an input. I create an input and then feed
        # zeros to the network. Ugly, but it's the same as disabling those weights.
        # Furthermore, Keras LSTM input=output, so we cannot produce more than SUBPOLICIES
        # outputs. This is not desirable, since the paper produces 25 subpolicies in the
        # end.
	pass

    def fit(self, mem_softmaxes, mem_accuracies):
	pass

    def predict(self, size, X):
        dummy_input = np.zeros((1, size, 1), np.float32)
        softmaxes = self.model.predict(dummy_input)
        # convert softmaxes into subpolicies
	print ("SM", softmaxes)
        subpolicies = []
        for i in range(SUBPOLICIES):
            operations = []
            for j in range(SUBPOLICY_OPS):
                op = softmaxes[j*3:(j+1)*3]
		print ("OP", op)
                op = [o[0, i, :] for o in op]
		print ("OP2", op)
                operations.append(Operation(X, *op))
            subpolicies.append(Subpolicy(*operations))
        return softmaxes, subpolicies

# generator
def autoaugment(subpolicies, X, y, child_batch_size):
    while True:
        ix = np.arange(len(X))
        np.random.shuffle(ix)
        for i in range(len(X) // child_batch_size):
            _ix = ix[i*child_batch_size:(i+1)*child_batch_size]
            _X = X[_ix]
            _y = y[_ix]
            subpolicy = np.random.choice(subpolicies)
            _X = subpolicy(_X)
            _X = _X.astype(np.float32) / 255
            yield _X, _y
