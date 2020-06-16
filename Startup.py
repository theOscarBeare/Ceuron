import numpy as np


Iris = [0, 1, 0]
IrisTargets = [1, 1, 0]

# the "build" function parameters (build is used for supervised learning (competitive to come)):
# Perceptron will call to the perceptron function
# LM is the learning method, Split or cross are being developed
# TF, you can choose the transfer function
# Architecture, you can determine Feed Forward or Multi layer, Others to come
# NoEpoch, you can choose the number of epochs the program will run, "Default" in the future will look at stopping...
# ...conditions
# Data inputs are targets need to be separated


def build(perceptron, TF, LM, Architecture, NoEpoch, dataInput, targets):
    Per = perceptron


build(perceptron="perceptron", TF="ReLu", LM= "split", Architecture="FFNN", NoEpoch=100, dataInput=Iris, targets=IrisTargets)
