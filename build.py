import numpy as np
from ADALINE import ADALINE1_9, ADALINE_Minus_Cross
from FFNN import FeedForward
from MLFFNN import BackPropagation
import pandas as pd

# setting the simple problem of the boolean AND problem with bias input. Used for the buildFFNN example
AND = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]])

# targets for the boolean AND problem. Used for the buildFFNN example
ANDTargets = np.array([0, 0, 0, 1, 0])

# Receiving the Iris dataset from an excel file using pandas
IrisDataInput = pd.read_excel("IrisDataInput.xlsx")

# Receiving the Iris targets from an excel file using pandas
IrisDataTarget = pd.read_excel("IrisDataTarget.xlsx")

# A Vector representation of a one bit grid that is 4x3
ExampleADALINEDataONE = np.array([1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1])

# example of determining which ADALINE network to use
whichADALINE = "NUMBER"

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

# Builds a Feed forward neural network, used for 2d linearly separable methods such as the boolean AND problem
def buildFFNN(TF, NoEpoch, dataInput, targets):

    for i in range(NoEpoch):
        FeedForward.feed_forward(TF, dataInput, targets)
        print("this is epoch number ", i+1)

# Builds a Multi layer feed forward neural network, this method uses back propagation and can have x number of hidden..
# .. layers
def buildMLFFNN(TF, NoEpoch, dataInput, targets):

    print(IrisDataInput, IrisDataTarget)

    for i in range(NoEpoch):

        print("this is epoch number ", i+1)

def buildADALINE(targetData, ErrorTolerance, whichADALINE):

    print("This is a simulation of an ADALINE Neural Network")
    Output = []

    if whichADALINE == "NUMBER":
        Output = ADALINE1_9.ADALINENetwork(targetData, ErrorTolerance)
    elif whichADALINE == "CROSSMINUS":
        Output = ADALINE_Minus_Cross.ADALINENetwork(targetData, ErrorTolerance)
    else:
        print("You have entered an invalid ADALINE set up")

    print(Output)


# Example of how the buildFFNN function can be used to build a feed forward neural network
buildFFNN("ReLu", 20, AND, ANDTargets)

# Example of how the buildMLFFNN function can be used to build a neural network to classify the iris dataset
buildMLFFNN("ReLu", 20, IrisDataInput, IrisDataTarget)

# Example of the ADALINE network
buildADALINE(ExampleADALINEDataONE, 2, whichADALINE)

