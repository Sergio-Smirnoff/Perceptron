import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class NonLinearPerceptron:
    """
    Perceptrón no lineal de una neurona con activación tanh.
    Escala automáticamente y a su vez desescala las salidas.
    """

    def __init__(self, learning_rate=0.001, epochs=2000, epsilon=1e-6, beta=1.0, random_seed=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.beta = beta
        self.random_seed = random_seed

        self.weights = None
        self.bias = None
        self.history_scaled = []
        self.history_real = []
        self._yscaler = None
        self.rng = np.random.RandomState(random_seed)
        print("seed: {random_seed}".format(random_seed=random_seed))

    def activation_function(self, x):
        """Función de activación Tangente Hiperbólica."""
        return np.tanh(self.beta * x)

    def train(self, X, y, verbose=False):
        """Entrena el perceptrón usando el conjunto de datos (X, y)."""
        n_samples, n_features = X.shape

        self.weights = self.rng.randn(n_features)
        self.bias = self.rng.randn()

        # Escala los valores de salida 'y' al rango [-1, 1] que maneja tanh
        self._yscaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = self._yscaler.fit_transform(y.reshape(-1, 1)).ravel()

        prev_mse_scaled = np.inf
        pbar = tqdm(range(self.epochs), disable=not verbose)

        for epoch in pbar:
            shuffled_indices = self.rng.permutation(n_samples)
            for i in shuffled_indices:
                xi = X[i]
                target = y_scaled[i]

                linear_output = np.dot(self.weights, xi) + self.bias
                prediction = self.activation_function(linear_output)

                error = target - prediction
                derivative = self.beta * (1 - prediction ** 2)
                delta = error * derivative

                self.weights += self.learning_rate * delta * xi
                self.bias += self.learning_rate * delta

            y_pred_scaled = self.activation_function(np.dot(X, self.weights) + self.bias)
            mse_scaled = np.mean((y_scaled - y_pred_scaled) ** 2)
            self.history_scaled.append(mse_scaled)

            y_pred_real = self._yscaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            mse_real = np.mean((y - y_pred_real) ** 2)
            self.history_real.append(mse_real)

            if verbose:
                pbar.set_description(f"Epoch {epoch + 1}, MSE(real): {mse_real:.6f}")

            if abs(prev_mse_scaled - mse_scaled) < self.epsilon:
                if verbose:
                    print(f"Convergencia alcanzada en epoch {epoch + 1}")
                break
            prev_mse_scaled = mse_scaled

        return self

    def predict(self, X):
        """Predice valores para un conjunto de entradas X."""
        X = np.asarray(X, dtype=float)
        single_input = X.ndim == 1
        X_in = X.reshape(1, -1) if single_input else X

        linear = np.dot(X_in, self.weights) + self.bias
        y_scaled = self.activation_function(linear)

        y_real = self._yscaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        return float(y_real[0]) if single_input else y_real





