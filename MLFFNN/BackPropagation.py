import numpy as np


def forwardPass(DataInput, Targets):
    pass

def backpropagation(weights1, weights2, input, layer1, output, y):
    # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    d_weights2 = np.dot(layer1.T, (2 * (y - output) * np.sigmoid_derivative(output)))
    d_weights1 = np.dot(input.T, (np.dot(2 * (y - output) * np.sigmoid_derivative(output),
                                              weights2.T) * np.sigmoid_derivative(layer1)))

    # update the weights with the derivative (slope) of the loss function
    weights1 += d_weights1
    weights2 += d_weights2
