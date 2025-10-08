import json
from SimplePerceptron import SimplePerceptron
import numpy as np
import random
from tqdm import tqdm # Para la barra de progreso

def activation_function(x:float) -> float:
    return x

class LinearPerceptron(SimplePerceptron):
    def __init__(self, learning_rate:float, epochs:int=100, epsilon:float=0.01, threshold:float=0.5):
        super().__init__(learning_rate, epochs, epsilon)
        self.threshold = threshold

    def train(self, X: np.ndarray, z: np.array):
        """Train linear perceptron with given training set and expected outputs.

        Args:
            X multidimensional array: Input variables.
            z array: Expected outputs.
        """
        log_file = open("training_log_lin.txt", "w")  
        self.weights = [ random.uniform(-0.5,0.5) for _ in range(len(X[0])) ]
        self.bias = random.uniform(-0.5, 0.5)
        for epoch in tqdm(range(self.epochs), desc="Training..."):
            sum_squared_error = 0.0
            
            log_file.write(",".join(f"{w:.4f}" for w in self.weights) + f",{self.bias:.4f},")

            for i, x in enumerate(X):
                linear_output = np.dot(x, self.weights) + self.bias
                output = activation_function(linear_output)

                error = z[i] - output
                sum_squared_error += error**2

                # update
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * x[j]
                self.bias += self.learning_rate * error

            mean_squared_error = sum_squared_error / len(X)
            log_file.write(f"{mean_squared_error}\n")

            if mean_squared_error < self.epsilon:
                print(f"Convergence reached at epoch {epoch+1}")
                break

        print(f"Training finished")
        print(f"Bias={self.bias}")
        log_file.close()

    def predict(self, X: np.ndarray) -> float:
        """Predict output for given input vector.
        Args:
            x array: Input vector.  
        Returns:
            bool: Predicted class (True or False).
        """ 
        return np.dot(X, self.weights) + self.bias