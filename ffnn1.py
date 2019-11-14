"""
FFNN1: A simple 1-layer feed-forward back-propagation neural network
"""

import numpy as np
import sys
import time


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative to the Sigmoid function"""
    return x * (1 - x)


class FFNN1:
    """1-layer feed-forward back-propagation neural network"""

    def __init__(self):
        np.random.seed(int(time.time()))

        # Weights: A 3-by-1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def get_current_weights(self):
        return self.synaptic_weights

    def train(self, inputs, outputs, ntimes):
        """
        Train the model to make accurate predictions while adjusting weights
        continually
        """
        for _ in range(ntimes):
            # Siphon the training data via the neuron
            thoughts = self.think(inputs)

            # Compute the error rate for back-propagation
            error_rate = outputs - thoughts

            # Adjust the weights
            self.synaptic_weights += np.dot(
                inputs.T, error_rate * sigmoid_derivative(thoughts)
            )

    def think(self, inputs):
        """Pass the inputs via the neuron to get output"""
        return sigmoid(np.dot(inputs.astype(float), self.synaptic_weights))


def main():
    nn = FFNN1()

    print("Intial weights:")
    print(nn.get_current_weights())

    inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    print("Inputs:")
    print(inputs)

    outputs = np.array([[0, 1, 1, 0]]).T
    print("Outputs:")
    print(outputs)

    ntimes, *data = sys.argv[1:]
    assert len(data) == 3
    ntimes = int(ntimes)

    print("Training times:", ntimes)

    nn.train(inputs, outputs, ntimes)
    print("Current weights:")
    print(nn.get_current_weights())

    print("User inputs:", *data)
    print("Answer:")
    print(nn.think(np.array(data)))


if __name__ == "__main__":
    main()
