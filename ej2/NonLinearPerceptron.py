# NonLinearPerceptron.py
import numpy as np
import random
from tqdm import tqdm

def scale_array(array):
    array = np.asarray(array, dtype=float)
    array_min = np.min(array, axis=0)
    array_max = np.max(array, axis=0)
    denom = (array_max - array_min)
    denom[denom == 0] = 1e-9
    scaled = 2 * (array - array_min) / denom -1
    return scaled, array_min, array_max

def descale_y(y_scaled, y_min, y_max):
    return ((y_scaled +1) / 2) * (y_max - y_min) + y_min

def activation_function(x, beta=1.0):
    return np.tanh(beta * x)

def activation_derivative(output, beta=1.0):
    # d/dx tanh(βx) = β * (1 - tanh^2(βx))
    return beta * (1.0 - output**2)

class NonLinearPerceptron:
    def __init__(self, learning_rate=0.01, epochs=100, epsilon=1e-36, beta=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.beta = beta
        self.weights = None
        self.bias = None
        # scalers (set on train)
        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None
        self.history = []

    def train(self, X, y, verbose=False):
        log_file = open("training_log_nonlin.txt", "a")
        if self.X_min is None:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)

            X_scaled, self.X_min, self.X_max = scale_array(X)
            y_scaled, self.y_min, self.y_max = scale_array(y.reshape(-1, 1))
            y_scaled = y_scaled.flatten()

            self.X_scaled = X_scaled
            self.y_scaled = y_scaled
            n_features = X.shape[1]
            rng = np.random.RandomState()
            self.weights = rng.uniform(-0.5, 0.5, size=n_features)
            self.bias = float(rng.uniform(-0.5, 0.5))

            self.history = []
        else:
            X_scaled = self.X_scaled
            y_scaled = self.y_scaled
        for epoch in tqdm(range(self.epochs), desc="Training", disable=not verbose):
            log_file.write(",".join(f"{w:.4f}" for w in self.weights) + f",{self.bias:.4f},")
            mse = 0.0
            # online / SGD update
            for xi, yi in zip(X_scaled, y_scaled):
                linear_output = np.dot(xi, self.weights) + self.bias
                output = activation_function(linear_output, self.beta)

                error = yi - output
                delta = error * activation_derivative(output, self.beta)

                # updates
                self.weights += self.learning_rate * delta * xi
                self.bias += self.learning_rate * delta

                mse += error**2

            mse /= len(X_scaled)
            self.history.append(mse)
            log_file.write(f"{mse}\n")

            if epoch % max(1, self.epochs // 10) == 0 and verbose:
                print(f"Epoch {epoch+1}/{self.epochs} MSE={mse:.6f}")

            if mse < self.epsilon:
                if verbose:
                    print(f"Converged at epoch {epoch+1} (mse {mse:.6g})")
                break

        if verbose:
            print(f"Training finished. Bias final: {self.bias:.6f}")

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        single = False
        if x.ndim == 1:
            single = True

        denom = (self.X_max - self.X_min)
        denom[denom == 0] = 1e-9
        x_scaled = 2 * (x - self.X_min) / denom - 1

        linear = x_scaled.dot(self.weights) + self.bias
        y_scaled = activation_function(linear, self.beta)
        y_pred = descale_y(y_scaled, self.y_min, self.y_max)

        y_pred = y_pred.flatten()
        return y_pred[0] if single else y_pred


