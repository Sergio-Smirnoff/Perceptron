import json
from SimplePerceptron import SimplePerceptron
import numpy as np

def activation_function(x:float) -> float:
    return x

class LinearPerceptron(SimplePerceptron):
    def __init__(self, learning_rate:float, epochs:int=100, epsilon:float=0.0):
        super().__init__(learning_rate, epochs, epsilon)
        self._step_activation_function = activation_function


    def train(self, X: np.ndarray, z: np.array):
        """Train linear perceptron with given training set and expected outputs.

        Args:
            X multidimensional array: Input variables.
            z array: Expected outputs.
        """

        for epoch in range(self.epochs):
            #mu es la variable de estudio, osea x, y, z, etc, los diferentes ejes del grafico
            #x_mu es el valor de la variable mu en esa muestra
            for variable, variable_value in enumerate(X):
                #1. Calculate weighted sum
                sum = np.dot(variable_value, self.weights) + self.bias

                #2. Compute activation function
                output = self._step_activation_function(sum)
                #3.Update weights and bias
                #4. Calculate error


        convergence = True if mean_squared_error < self.epsilon else False
        if convergence: break