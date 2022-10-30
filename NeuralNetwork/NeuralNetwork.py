#  -*- coding: utf-8 -*-
__author__ = "kubik.augustyn@post.cz"

import numpy as np
from Layer import Layer
import cv2


class NeuralNetwork:
    layers = []

    # Create the Neural Network
    def __init__(self, layerSizes):
        self.layers = []
        for i in range(len(layerSizes) - 1):
            self.layers.append(Layer(layerSizes[i], layerSizes[i + 1]))

    # Run the inputs through the network to calculate the outputs
    def CalculateOutputs(self, inputs):
        for layer in self.layers:
            inputs = layer.CalculateOutputs(inputs)
        return inputs

    # Run the inputs through the network ad calculate which output node has the highest value
    def Classify(self, inputs):
        outputs = self.CalculateOutputs(inputs)
        return np.argmax(outputs)

    def Visualise(self, graph, graphX, graphY, graphW, graphH):
        predictedClass = self.Classify([graphX / graphW, graphY / graphH])

        if predictedClass == 0:  # Safe
            graph[graphY, graphX] = (0, 255, 0)
        else:  # Poisonous
            graph[graphY, graphX] = (0, 0, 255)


if __name__ == '__main__':
    net = NeuralNetwork([2, 2])


    def nothing(x):
        pass


    # Create a black image, a window
    settings = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('Settings')
    # create trackbars for color change
    cv2.createTrackbar('Bias 1', 'Settings', 128, 255, nothing)
    cv2.createTrackbar('Bias 2', 'Settings', 128, 255, nothing)

    cv2.createTrackbar('Weight 1 1', 'Settings', 128, 255, nothing)
    cv2.createTrackbar('Weight 1 2', 'Settings', 128, 255, nothing)
    cv2.createTrackbar('Weight 2 1', 'Settings', 128, 255, nothing)
    cv2.createTrackbar('Weight 2 2', 'Settings', 128, 255, nothing)

    while True:
        w = 128
        h = 128
        img = np.zeros((h, w, 3))
        net.layers[0].biases[0] = (cv2.getTrackbarPos('Bias 1', 'Settings') - 128) / 128
        net.layers[0].biases[1] = (cv2.getTrackbarPos('Bias 2', 'Settings') - 128) / 128
        print(net.layers[0].biases)

        net.layers[0].weights[0, 0] = (cv2.getTrackbarPos('Weight 1 1', 'Settings') - 128) / 128
        net.layers[0].weights[0, 1] = (cv2.getTrackbarPos('Weight 1 2', 'Settings') - 128) / 128
        net.layers[0].weights[1, 0] = (cv2.getTrackbarPos('Weight 2 1', 'Settings') - 128) / 128
        net.layers[0].weights[1, 1] = (cv2.getTrackbarPos('Weight 2 2', 'Settings') - 128) / 128
        for y in range(h):
            for x in range(w):
                net.Visualise(img, x, y, w, h)
        cv2.imshow('Graph', img)
        cv2.imshow('Settings', settings)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
