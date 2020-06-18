from FFNN import FeedForward

# setting the simple problem of the boolean AND problem with bias input
AND = [[1, 0, 0],
       [1, 0, 1],
       [1, 1, 0],
       [1, 1, 1]]

# targets for the boolean AND problem
ANDTargets = [0, 0, 0, 1]

# the "build" function parameters (build is used for supervised learning (competitive to come)):
# Perceptron will call to the perceptron function
# LM is the learning method, Split or cross are being developed
# TF, you can choose the transfer function
# Architecture, you can determine Feed Forward or Multi layer, Others to come
# NoEpoch, you can choose the number of epochs the program will run, "Default" in the future will look at stopping...
# ...conditions
# Data inputs are targets need to be separated


def build(TF, LM, Architecture, NoEpoch, dataInput, targets):
    for i in NoEpoch:
        weights = FeedForward.feed_forward(TF, dataInput, targets)
        print(weights)





