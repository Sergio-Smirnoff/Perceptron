import json
from LinearPerceptron import LinearPerceptron
from SimplePerceptron import SimplePerceptron
import numpy as np
import random
from tqdm import tqdm # Para la barra de progreso

def activation_function(x:float) -> float:
    with open("../params.json", "r") as f:
        params = json.load(f)
    beta = params.get("non_linear_beta", 1.0)
    return 1 / (1 + np.exp(-2 * beta * x))

def activation_derivative(output, beta=1.0):
    return 2 * beta * output * (1 - output)

#TODO verify that normalization is correct. other options: minmax scaling, z-score normalization, tu vieja
def scale_data(X, y):
    X = [scale_array(row) for row in X]
    y = scale_array(y)
    return X, y

def scale_array(array):
    """Scale array with MINMAX SCALING.
    Hardcoded for 1D array. >:D"""
    array_min = np.min(array)
    array_max = np.max(array)
    return (array - array_min) / (array_max - array_min)

def descale_data(y, y_min, y_max):
    """Descale data with MINMAX SCALING.
    Hardcoded for 1 output variable. >:D"""
    return y * (y_max - y_min) + y_min

class NonLinearPerceptron(LinearPerceptron):
    def __init__(self, learning_rate:float, epochs:int=100, epsilon:float=0.01):
        super().__init__(learning_rate, epochs, epsilon)
        # self._step_activation_function = activation_function
    def train(self, X: np.ndarray, z: np.array):
        """Train linear perceptron with given training set and expected outputs.

        Args:
            X multidimensional array: Input variables.
            z array: Expected outputs.
        """
        log_file = open("training_log.txt", "w")  
        self.weights = [ random.uniform(-0.5,0.5) for _ in range(len(X[0])) ]
        self.bias = random.uniform(-0.5, 0.5)

        #======== SCALE DATA ========
        #MEGA IMPORTANTE
        X, z = scale_data(X, z)

        for epoch in tqdm(range(self.epochs), desc="Training..."):
            sum_squared_error = 0.0

            for i, x in enumerate(X):
                linear_output = np.dot(x, self.weights) + self.bias
                output = activation_function(linear_output)

                error = z[i] - output
                sum_squared_error += error**2

                # update
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * activation_derivative(output) * x[j]
                self.bias += self.learning_rate * error * activation_derivative(output)

                log_file.write(",".join(f"{w:.4f}" for w in self.weights) + f",{self.bias:.4f}\n")

            mean_squared_error = sum_squared_error / len(X)
            print(f'MSE at epoch {epoch+1}: {mean_squared_error}')

            if mean_squared_error < self.epsilon:
                print(f"Convergence reached at epoch {epoch+1}")
                break

        print(f"Training finished")
        print(f"Bias={self.bias}")
        log_file.close()

    def predict(self, x: np.ndarray) -> float:
        """Predict output for given input vector.
        Args:
            x array: Input vector.  
        Returns:
            float: Predicted output.
        """
        
        """Predice para un único vector x de tamaño 3, usando los min/max del TRAIN."""

        xs = scale_array(x)         # usa min/max del TRAIN
        y_scaled = activation_function(np.dot(xs, self.weights) + self.bias)
        y_pred = descale_data(y_scaled, 0, 1)    # vuelve al rango original de y
        return float(y_pred)