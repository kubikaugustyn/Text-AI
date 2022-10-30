#  -*- coding: utf-8 -*-
__author__ = "kubik.augustyn@post.cz"

import numpy as np


class Layer:
    numNodesIn = None
    numNodesOut = None

    weights = []
    biases = []

    # Create the layer
    def __init__(self, numNodesIn, numNodesOut):
        self.numNodesIn = numNodesIn
        self.numNodesOut = numNodesOut

        self.weights = np.zeros((numNodesIn, numNodesOut))
        self.biases = np.zeros(numNodesOut)

    # Calculate the output of the layer
    def CalculateOutputs(self, inputs):
        weightedInputs = np.zeros(self.numNodesOut)

        for nodeOut in range(self.numNodesOut):
            weightedInput = self.biases[nodeOut]
            for nodeIn in range(self.numNodesIn):
                weightedInput += inputs[nodeIn] * self.weights[nodeIn][nodeOut]
            weightedInputs[nodeOut] = weightedInput

        return weightedInputs
