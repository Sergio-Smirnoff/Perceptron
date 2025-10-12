# NonLinearPerceptron.py
import numpy as np
from tqdm import tqdm

## Funciones auxiliares
def scale_array(array):
    """Escala array de 1D!!!!!!!!
    Output en (-1, 1)
    """
    array = np.asarray(array, dtype=float)
    array_min = np.min(array)
    array_max = np.max(array)
    denom = (array_max - array_min)
    # denom[denom == 0] = 1e-9
    #Xnew= (Xold - Xmin)/(Xmax - Xmin) * (newMax - newMin) + newMin
    if denom == 0:
        return np.zeros_like(array), array_min, array_max
    scaled = 2 * (array - array_min) / denom - 1
    return scaled, array_min, array_max


def descale_y(y_scaled, y_min, y_max):
    return ((y_scaled + 1) / 2) * (y_max - y_min) + y_min


def activation_function(x, beta=1.0):
    return np.tanh(beta * x)


def activation_derivative(output, beta=1.0):
    return beta * (1.0 - output ** 2)


## Clase del Perceptrón
class NonLinearPerceptron:
    def __init__(self, learning_rate=0.01, epochs=100, epsilon=1e-36, beta=1.0):
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
        self.history = []

    def train(self, X, y, X_test=None, y_test=None, train_log_file=None, test_log_file=None, global_scaling_params=None, verbose=False):
        # Escalado por columnas
        if global_scaling_params:
            self.X_min, self.X_max, self.y_min, self.y_max = global_scaling_params
        else:
            self.X_min = np.min(X, axis=0)
            self.X_max = np.max(X, axis=0)
            _, self.y_min, self.y_max = scale_array(y)

        denom = self.X_max - self.X_min
        denom[denom == 0] = 1e-9
        X_scaled = 2 * (X - self.X_min) / denom - 1
        
        y_denom = self.y_max - self.y_min
        if y_denom == 0:
            y_scaled = np.zeros_like(y)
        else:
            y_scaled = 2 * (y - self.y_min) / y_denom - 1
        
        y_scaled = y_scaled.flatten()

        n_features = X.shape[1]
        rng = np.random.default_rng()
        self.weights = rng.uniform(-0.5, 0.5, size=n_features)
        self.bias = rng.uniform(-0.5, 0.5)
        self.history = []

        # Entrenamiento online
        for epoch in range(self.epochs):
            mse_scaled = 0.0
            outputs = []
            for xi, yi in zip(X_scaled, y_scaled):
                output = activation_function(np.dot(xi, self.weights) + self.bias, self.beta)
                outputs.append(output)
                error = yi - output
                delta = error * activation_derivative(output, self.beta)
                self.weights += self.learning_rate * delta * xi
                self.bias += self.learning_rate * delta
                mse_scaled += error ** 2
            mse_scaled /= len(X_scaled)
            self.history.append(mse_scaled)

            mse = 0
            for yi, oi in zip(y, outputs):
                o = descale_y(oi, self.y_min, self.y_max)
                error = yi - o
                mse += error ** 2
            mse /= len(y)
            if train_log_file:
                train_log_file.write(f"{mse}\n")

            # Test
            if X_test is not None and y_test is not None and test_log_file:
                y_pred_test = np.array([self.predict(xi) for xi in X_test])
                test_mse = np.mean((y_test - y_pred_test) ** 2)
                test_log_file.write(f"{test_mse}\n")


    def predict(self, x):
        """
        Realiza una predicción para una entrada dada.
        x: Array de 3 elementos -> x=[x1, x2, x3]
        """

        #Xnew= (Xold - Xmin)/(Xmax - Xmin) * (newMax - newMin) + newMin
        # x1 = (2 * (x[0] - self.X_min[0]) / (self.X_max[0] - self.X_min[0])) - 1
        # x2 = (2 * (x[1] - self.X_min[1]) / (self.X_max[1] - self.X_min[1])) - 1
        # x3 = (2 * (x[2] - self.X_min[2]) / (self.X_max[2] - self.X_min[2])) - 1
        # x_scaled = np.array([x1, x2, x3])
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1e-9
        x_scaled = 2 * (x - self.X_min) / denom - 1

        linear = np.dot(x_scaled, self.weights) + self.bias
        y_scaled = activation_function(linear, self.beta)

        y_pred = descale_y(y_scaled, self.y_min, self.y_max)

        return y_pred


