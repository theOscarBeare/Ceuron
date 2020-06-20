import numpy as np
from FFNN import FeedForward

# setting the simple problem of the boolean AND problem with bias input. Used for the buildFFNN example
AND = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]])

# targets for the boolean AND problem. Used for the buildFFNN example
ANDTargets = np.array([0, 0, 0, 1, 0])

#######################################################################################################################
# the "build" function parameters (build is used for supervised learning (competitive to come)):
# Perceptron will call to the perceptron function
# LM is the learning method, Split or cross are being developed
# TF, you can choose the transfer function
# Architecture, you can determine Feed Forward or Multi layer, Others to come
# NoEpoch, you can choose the number of epochs the program will run, "Default" in the future will look at stopping...
# ...conditions
# Data inputs are targets need to be separated
#######################################################################################################################

# Builds a Feed forward neural network, used for 2d linearly separable methods
def buildFFNN(TF, NoEpoch, dataInput, targets):

    for i in range(NoEpoch):
        weights = FeedForward.feed_forward(TF, dataInput, targets)
        print(weights)

# Builds a Multi layer feed forward neural network, this method uses back propagation and can have x number of hidden..
# .. layers
def buildMLFFNN(TF, NoEpoch, dataInput, targets, LM, NoHiddenLayers):
    pass


# Example of how the buildFFNN function can be used to build a feed forward neural network
buildFFNN("ReLu", 20, AND, ANDTargets)



