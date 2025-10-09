# NonLinearPerceptron.py
import numpy as np
from tqdm import tqdm


## Funciones auxiliares (sin cambios)
def scale_array(array):
    array = np.asarray(array, dtype=float)
    array_min = np.min(array, axis=0)
    array_max = np.max(array, axis=0)
    denom = (array_max - array_min)
    denom[denom == 0] = 1e-9
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

    # El método 'train' está ahora correctamente indentado y modificado
    def train(self, X, y, X_test=None, y_test=None, train_log_file=None, test_log_file=None, verbose=False):
        # Escalar datos de entrenamiento y guardar escaladores
        X_scaled, self.X_min, self.X_max = scale_array(X)
        y_scaled, self.y_min, self.y_max = scale_array(y.reshape(-1, 1))
        y_scaled = y_scaled.flatten()

        # Inicializar pesos y bias
        n_features = X.shape[1]
        rng = np.random.RandomState()
        self.weights = rng.uniform(-0.5, 0.5, size=n_features)
        self.bias = float(rng.uniform(-0.5, 0.5))
        self.history = []
<<<<<<< HEAD:ej2/NonLinearPerceptron.py

        # Bucle de entrenamiento principal
        for epoch in tqdm(range(self.epochs), desc="Training Fold", disable=not verbose, leave=False):
            # Lógica de entrenamiento para una época
            mse_scaled = 0.0
=======
        log_file = open("training_log_nonlin.txt", "w")

        for epoch in tqdm(range(self.epochs), desc="Training", disable=not verbose):
            
            mse = 0.0
            # online / SGD update
>>>>>>> multicapa:NonLinearPerceptron.py
            for xi, yi in zip(X_scaled, y_scaled):
                linear_output = np.dot(xi, self.weights) + self.bias
                output = activation_function(linear_output, self.beta)
                error = yi - output
                delta = error * activation_derivative(output, self.beta)
                self.weights += self.learning_rate * delta * xi
                self.bias += self.learning_rate * delta
                mse_scaled += error ** 2

            mse_scaled /= len(X_scaled)
            self.history.append(mse_scaled)

<<<<<<< HEAD:ej2/NonLinearPerceptron.py
            # Calcular MSE sobre datos originales sin escalar
            y_pred_train = self.predict(X)
            train_mse = np.mean((y - y_pred_train) ** 2)

            # Escribir en el log de entrenamiento si se pasa
            if train_log_file:
                train_log_file.write(f"{train_mse}\n")
=======
            mse /= len(X_scaled)
            self.history.append(mse)
            log_file.write(f"{mse}\n")
            if epoch % max(1, self.epochs // 10) == 0 and verbose:
                print(f"Epoch {epoch+1}/{self.epochs} MSE={mse:.6f}")
>>>>>>> multicapa:NonLinearPerceptron.py

            # Calcular y escribir en el log de test si se pasa
            if X_test is not None and y_test is not None and test_log_file:
                y_pred_test = self.predict(X_test)
                test_mse = np.mean((y_test - y_pred_test) ** 2)
                test_log_file.write(f"{test_mse}\n")

            # Condición de corte
            if mse_scaled < self.epsilon:
                if verbose:
                    print(f"Convergencia alcanzada en época {epoch + 1}")
                break
<<<<<<< HEAD:ej2/NonLinearPerceptron.py
=======
        log_file.write(",".join(f"{w:.4f}" for w in self.weights) + f",{self.bias:.4f},")
        if verbose:
            print(f"Training finished. Bias final: {self.bias:.6f}")
>>>>>>> multicapa:NonLinearPerceptron.py

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        is_single_item = x.ndim == 1

        x_input = x.reshape(1, -1) if is_single_item else x

        denom = (self.X_max - self.X_min)
        denom[denom == 0] = 1e-9
        x_scaled = 2 * (x_input - self.X_min) / denom - 1

        linear = np.dot(x_scaled, self.weights) + self.bias
        y_scaled = activation_function(linear, self.beta)
        y_pred = descale_y(y_scaled, self.y_min, self.y_max)

        return y_pred[0] if is_single_item else y_pred


