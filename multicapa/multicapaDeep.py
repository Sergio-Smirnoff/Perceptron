import logging as log
import numpy as np
import random
from tqdm import tqdm

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MulticapaDeep:
    """
    Perceptrón multicapa con número arbitrario de capas ocultas.
    Soporta múltiples capas y diferentes tamaños por capa.
    """

    def __init__(self, layer_sizes, learning_rate=0.1, epochs=1000, epsilon=0.001):
        """
        Args:
            layer_sizes (list): Lista con tamaño de cada capa.
                               Ejemplo: [2, 4, 3, 1] = entrada(2) -> oculta1(4) -> oculta2(3) -> salida(1)
            learning_rate (float): Tasa de aprendizaje
            epochs (int): Número de épocas
            epsilon (float): Error mínimo para convergencia
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon

        # Inicializar pesos y biases para cada capa
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # Xavier initialization
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.array([
                [random.uniform(-limit, limit) for _ in range(layer_sizes[i+1])]
                for _ in range(layer_sizes[i])
            ])
            b = np.array([random.uniform(-limit, limit) for _ in range(layer_sizes[i+1])])

            self.weights.append(w)
            self.biases.append(b)

        log.info(f"Red neuronal inicializada: {' -> '.join(map(str, layer_sizes))}")

    def _sigmoid(self, x):
        """Función de activación sigmoide"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, output):
        """Derivada de sigmoid (usando salida ya activada)"""
        return output * (1 - output)

    def forward_pass(self, x):
        """
        Propagación hacia adelante por TODAS las capas.

        Args:
            x (np.ndarray): Vector de entrada

        Returns:
            list: Lista con salidas de cada capa [h1, h2, ..., hN, y]
        """
        activations = [x]  # Guardar activación de cada capa (empezando por entrada)

        current_input = x
        for i in range(self.num_layers - 1):
            # z = W * a + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            # a = σ(z)
            activation = self._sigmoid(z)
            activations.append(activation)
            current_input = activation

        return activations

    def backward_pass(self, x, z_expected, activations):
        """
        Propagación hacia atrás (backpropagation) por TODAS las capas.
        Usa regla de la cadena para calcular gradientes.

        Args:
            x (np.ndarray): Vector de entrada original

            z_expected (float): Valor esperado
                z = [expected_l1(), expected_l2()]

            activations (list): Salidas de cada capa (del forward pass)
                    act = [activation_l1(), activation_l2()]

        Returns:
            tuple: (gradientes_pesos, gradientes_biases, error)
        """
        # Inicializar listas para gradientes
        grad_weights = [None] * (self.num_layers - 1)
        grad_biases = [None] * (self.num_layers - 1)

        # Paso 1: Error en la capa de salida
        output = activations[-1]
        # Asegurar que sean escalares si la salida es de tamaño 1
        if isinstance(output, np.ndarray) and len(output) == 1:
            output = output[0]

        expected_l2 = z_expected[-1] #TODO check
        error = expected_l2 - output

        # Delta de la capa de salida
        delta = error * self._sigmoid_derivative(output)

        # Paso 2: Backpropagation - iterar hacia atrás desde la última capa
        for layer in range(self.num_layers - 2, -1, -1):
            # Activación de la capa anterior
            prev_activation = activations[layer]

            # Calcular gradientes para esta capa
            # ∂E/∂W = δ * a_(prev)^T
            # Asegurar que delta sea array para operaciones vectoriales
            delta_array = np.atleast_1d(delta)
            grad_weights[layer] = np.outer(prev_activation, delta_array)
            # ∂E/∂b = δ
            grad_biases[layer] = delta_array

            # Si no es la primera capa, propagar error hacia atrás
            if layer > 0:
                # δ_(l-1) = (W_l^T · δ_l) ⊙ σ'(a_(l-1))
                # Propagar error multiplicando por pesos transpuestos
                delta = np.dot(self.weights[layer], delta_array) * self._sigmoid_derivative(prev_activation)

        return grad_weights, grad_biases, error

    def update_weights(self, grad_weights, grad_biases):
        """
        Actualizar pesos y biases de TODAS las capas.

        Args:
            grad_weights (list): Lista de gradientes de pesos por capa
            grad_biases (list): Lista de gradientes de biases por capa
        """
        for i in range(self.num_layers - 1):
            self.weights[i] += self.learning_rate * grad_weights[i]
            self.biases[i] += self.learning_rate * grad_biases[i]

    def train(self, X, z):
        """
        Entrenar la red neuronal multicapa.

        Args:
            X (np.ndarray): Conjunto de entradas
            z (np.ndarray): Conjunto de salidas esperadas
        """
        log.info(f"Iniciando entrenamiento: {self.epochs} épocas, lr={self.learning_rate}")

        # Normalizar salidas de [-1, 1] a [0, 1] para sigmoid
        z_normalized = (z + 1) / 2

        convergence = False

        for epoch in tqdm(range(self.epochs), desc="Entrenando"):
            sum_squared_error = 0.0

            # Acumular gradientes para batch update
            batch_grad_w = [np.zeros_like(w) for w in self.weights]
            batch_grad_b = [np.zeros_like(b) for b in self.biases]

            for x_i, z_i in zip(X, z_normalized):
                # 1. Forward pass
                activations = self.forward_pass(x_i)

                # 2. Backward pass
                grad_w, grad_b, error = self.backward_pass(x_i, z_i, activations)

                # 3. Acumular gradientes
                for i in range(self.num_layers - 1):
                    batch_grad_w[i] += grad_w[i]
                    batch_grad_b[i] += grad_b[i]

                # Asegurar que error sea escalar para la suma
                error_scalar = float(error) if isinstance(error, (np.ndarray, np.generic)) else error
                sum_squared_error += error_scalar**2

            # 4. Actualizar pesos con gradientes acumulados
            self.update_weights(batch_grad_w, batch_grad_b)

            # Calcular MSE
            mean_squared_error = sum_squared_error / len(X)

            # Verificar convergencia
            if mean_squared_error < self.epsilon:
                convergence = True
                log.info(f"Convergencia alcanzada en época {epoch + 1} con MSE={mean_squared_error:.6f}")
                break

            if (epoch + 1) % 1000 == 0:
                log.info(f"Época {epoch + 1}/{self.epochs} - MSE: {mean_squared_error:.6f}")

        final_mse = sum_squared_error / len(X)
        log.info(f"Entrenamiento finalizado después de {epoch + 1} épocas")
        log.info(f"Convergencia: {'ALCANZADA' if convergence else 'NO ALCANZADA'}")
        log.info(f"Error cuadrático medio final: {final_mse:.6f}")

    def predict(self, x):
        """
        Realizar predicción con la red entrenada.

        Args:
            x (np.ndarray): Vector de entrada

        Returns:
            int: Predicción (1 o -1)
        """
        activations = self.forward_pass(x)
        final_output = activations[-1]

        # Si la salida es escalar, tomar el primer elemento
        if isinstance(final_output, np.ndarray):
            final_output = final_output[0] if len(final_output) == 1 else final_output

        # Convertir de [0,1] a [-1,1]
        return 1 if final_output >= 0.5 else -1


# Función principal para probar
if __name__ == "__main__":
    # Datos de entrenamiento para XOR
    X = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1]
    ])
    z = np.array([-1, 1, 1, -1])

    print("\n" + "="*70)
    print("EJEMPLO 1: XOR con 2 capas ocultas")
    print("="*70)
    # Arquitectura: 2 entrada -> 3 oculta1 -> 2 oculta2 -> 1 salida
    mlp1 = MulticapaDeep(
        layer_sizes=[2, 3, 2, 1],
        learning_rate=1.0,
        epochs=10000,
        epsilon=0.001
    )
    mlp1.train(X, z)

    # Probar predicciones
    log.info("\n=== PRUEBAS DE PREDICCIÓN ===")
    for x_i, z_i in zip(X, z):
        prediction = mlp1.predict(x_i)
        log.info(f"Entrada: {x_i} -> Predicción: {prediction:2d}, Esperado: {z_i:2d} {'✓' if prediction == z_i else '✗'}")

    print("\n" + "="*70)
    print("EJEMPLO 2: XOR con 3 capas ocultas")
    print("="*70)
    # Arquitectura: 2 entrada -> 4 oculta1 -> 3 oculta2 -> 2 oculta3 -> 1 salida
    mlp2 = MulticapaDeep(
        layer_sizes=[2, 4, 3, 2, 1],
        learning_rate=0.8,
        epochs=10000,
        epsilon=0.001
    )
    mlp2.train(X, z)

    # Probar predicciones
    log.info("\n=== PRUEBAS DE PREDICCIÓN ===")
    for x_i, z_i in zip(X, z):
        prediction = mlp2.predict(x_i)
        log.info(f"Entrada: {x_i} -> Predicción: {prediction:2d}, Esperado: {z_i:2d} {'✓' if prediction == z_i else '✗'}")
