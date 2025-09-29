import json
from SimplePerceptron import SimplePerceptron
import numpy as np
import random
from tqdm import tqdm # Para la barra de progreso

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
        log_file = open("training_log.txt", "w")  
        self.weights = [ random.uniform(-0.5,0.5) for _ in range(len(X[0])) ]
        self.bias = random.uniform(-0.5, 0.5)
        for _ in tqdm(range(self.epochs), desc="Training..."):
            sum_squared_error = 0.0
            #X = (x1, x2, x3, x4,...)
            #x1 = (value1, value2), x2 = (value1', value 2'),  ....
            for variable, variable_value in enumerate(X):
                #1. Calculate weighted sum
                sum = np.dot(variable_value, self.weights) + self.bias

                #2. Compute activation function
                output = self._step_activation_function(sum)

                #3.Update weights and bias
                for w in range(len(self.weights)):

                    #Formula
                    #wi = wi + learn_rate * (zi - output) * xi * activation_func'
                    #activation_func = Id -> activation_func' = 1
                    self.weights[w] = self.weights[w] + self.learning_rate * (z[variable] - output) * variable_value[w]
                self.bias = self.bias + self.learning_rate * (z[variable] - output)
                
                log_file.write(",".join(f"{w:.4f}" for w in self.weights) + f",{self.bias:.4f}\n")


                #4. Calculate error
                error = z[variable] - output
                sum_squared_error += error**2
                mean_squared_error = sum_squared_error / 2
            convergence = True if mean_squared_error < self.epsilon else False
            if convergence: break
        print(f"Training finished")
        print(f"Convergence was {'reached' if convergence else 'not reached'}")
        print(f"Bias={self.bias}")
        log_file.close()