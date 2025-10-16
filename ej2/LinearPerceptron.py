import numpy as np
import random
from tqdm import tqdm

def activation_function(x:float) -> float:
    return x

class LinearPerceptron():
    def __init__(self, learning_rate:float, epochs:int=100, epsilon:float=0.01, threshold:float=0.5):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.epochs = epochs
        self.epsilon = epsilon
        self.threshold = threshold

    def train(self, X: np.ndarray, z: np.array):
        """Train linear perceptron with given training set and expected outputs.

        Args:
            X multidimensional array: Input variables.
            z array: Expected outputs.
        """
        num_features = X.shape[1]
        self.weights = np.array([random.uniform(-0.5, 0.5) for _ in range(num_features)])
        self.bias = random.uniform(-0.5, 0.5)
        errors_per_epoch = []
        predicts_per_epoch = []

        for epoch in tqdm(range(self.epochs), desc="Training..."):
            sum_squared_error = 0.0
            predicts = []

            for i, x in enumerate(X):
                linear_output = np.dot(x, self.weights) + self.bias

                output = activation_function(linear_output)
                predicts.append(output)

                error = z[i] - output

                sum_squared_error += error**2

                # update
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * x[j]
                self.bias += self.learning_rate * error

            mean_squared_error = sum_squared_error / len(X)
            errors_per_epoch.append(mean_squared_error)
            predicts_per_epoch.append(predicts)
            
            if mean_squared_error < self.epsilon:
                print(f"\nConvergencia alcanzada en la época {epoch+1} con un MSE de {mean_squared_error:.6f}")
                break
        
        print(f"\nEntrenamiento finalizado.")
        print(f"Pesos finales: {self.weights}")
        print(f"Bias final: {self.bias}")
        return errors_per_epoch, predicts_per_epoch

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias