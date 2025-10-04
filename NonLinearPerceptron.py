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
    """Scale data With MINMAX SCALING.
    Hardcoded for 3 input variables and 1 output variable. >:D"""
    x1_min=np.min(X[:, 0])
    x1_max=np.max(X[:, 0])
    x2_min=np.min(X[:, 1])
    x2_max=np.max(X[:, 1])
    x3_min=np.min(X[:, 2])
    x3_max=np.max(X[:, 2])
    y_min=np.min(y)
    y_max=np.max(y)
    X[:, 0] = (X[:, 0] - x1_min) / (x1_max - x1_min)
    X[:, 1] = (X[:, 1] - x2_min) / (x2_max - x2_min)
    X[:, 2] = (X[:, 2] - x3_min) / (x3_max - x3_min)
    y = (y - y_min) / (y_max - y_min)
    return X, y

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
        X, _ = scale_data(x, np.array([0]))
        nonlinear_output = np.dot(X, self.weights[:-1]) + self.bias
        return descale_data(activation_function(nonlinear_output), 0, 1)