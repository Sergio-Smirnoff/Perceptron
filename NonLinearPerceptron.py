import numpy as np
import random
from tqdm import tqdm

def scale_array(array):
    array_min = np.min(array, axis=0)
    array_max = np.max(array, axis=0)
    scaled = (array - array_min) / (array_max - array_min)
    return scaled, array_min, array_max

def descale_y(y_scaled, y_min, y_max):
    return y_scaled * (y_max - y_min) + y_min

def activation_function(x, beta=1.0):
    return 1 / (1 + np.exp(-2 * beta * x))

def activation_derivative(output, beta=1.0):
    return 2 * beta * output * (1 - output)


class NonLinearPerceptron:
    def __init__(self, learning_rate=0.01, epochs=100, epsilon=0.01, beta=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.beta = beta
        self.weights = None
        self.bias = None

        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None

    def train(self, X, y):
        X_scaled, self.X_min, self.X_max = scale_array(X)
        y_min, y_max = np.min(y), np.max(y)
        self.y_min, self.y_max = y_min, y_max
        y_scaled = (y - y_min) / (y_max - y_min)  # escala 0-1 para cálculo de gradiente

        n_features = X.shape[1]
        self.weights = np.array([random.uniform(-0.5,0.5) for _ in range(n_features)])
        self.bias = random.uniform(-0.5, 0.5)

        for epoch in tqdm(range(self.epochs), desc="Training"):
            mse = 0.0
            for xi, yi in zip(X_scaled, y_scaled):
                linear_output = np.dot(xi, self.weights) + self.bias
                output = linear_output

                error = yi - output
                mse += error**2

                # actualizar pesos
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

            mse /= len(X_scaled)
            if mse < self.epsilon:
                print(f"Converged at epoch {epoch+1}")
                break

        print(f"Training finished. Bias final: {self.bias:.4f}")

    def predict(self, x):
        x_scaled = (x - self.X_min) / (self.X_max - self.X_min)
        y_scaled = np.dot(x_scaled, self.weights) + self.bias
        y_pred = descale_y(y_scaled, self.y_min, self.y_max)
        return y_pred
