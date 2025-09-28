# SimplePerceptron.py
import numpy as np
import random as random
from tqdm import tqdm # Para la barra de progreso

class SimplePerceptron:
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def _step_activation_function(self, x:float) -> int:
        return 1 if x >= 0 else -1
    
    def train(self, X:np.ndarray, z:np.ndarray, epochs:int=100, epsilon:float=0.0):
        """
        Args:
            epochs (int)
            epsilon (float)
            X (list list of int): input descriptors vector.
            z (list of int): expected outputs.
        """
        self.weights = [ random.uniform(-0.5,0.5) for _ in range(X.shape[1]) ]
        self.bias = random.uniform(-0.5, 0.5)
        for _ in tqdm(range(epochs), desc="Training..."):
            sum_squared_error = 0.0
            for x_idx, x_i in enumerate(X):
                # Calculate weighted sum
                sum = np.dot(x_i, self.weights) + self.bias

                # Compute activation
                output = self._step_activation_function(sum)

                # Update weights and bias
                for w_idx, w_i in enumerate(self.weights):
                    self.weights[w_idx] = w_i + self.learning_rate * (z[x_idx] - output) * x_i[w_idx]
                self.bias = self.bias + self.learning_rate * (z[x_idx] - output)

                # Calculate error
                error = z[x_idx] - output
                sum_squared_error += error**2

            mean_squared_error = sum_squared_error / 2
            convergence = True if mean_squared_error < epsilon else False
            if convergence: break