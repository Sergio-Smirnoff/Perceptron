import logging as log
import numpy as np
import random
from tqdm import tqdm

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MulticapaXOR:
    """
    Perceptrón multicapa para aprender la función XOR.
    Arquitectura: 2 entradas -> capa oculta (2 neuronas) -> 1 salida
    """

    def __init__(self, learning_rate: float, epochs: int = 100, epsilon: float = 0.0):
        """
        Args:
            learning_rate (float): Tasa de aprendizaje
            epochs (int): Número de épocas de entrenamiento
            epsilon (float): Error mínimo para convergencia
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon

        # Arquitectura de la red: 2 entradas, 2 neuronas ocultas, 1 salida
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 1

        # Inicialización de pesos aleatorios
        # Pesos capa de entrada a capa oculta (2x2 + bias)
        self.weights_input_hidden = np.array([
            [random.uniform(-0.5, 0.5) for _ in range(self.hidden_size)]
            for _ in range(self.input_size)
        ])
        self.bias_hidden = np.array([random.uniform(-0.5, 0.5) for _ in range(self.hidden_size)])

        # Pesos capa oculta a capa de salida (2x1 + bias)
        self.weights_hidden_output = np.array([
            random.uniform(-0.5, 0.5) for _ in range(self.hidden_size)
        ])
        self.bias_output = random.uniform(-0.5, 0.5)

        log.info(f"Red neuronal inicializada: {self.input_size} -> {self.hidden_size} -> {self.output_size}")

    def _sigmoid(self, x):
        """Función de activación sigmoide"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip para evitar overflow

    def _sigmoid_derivative(self, x):
        """Derivada de la función sigmoide"""
        return x * (1 - x)

    def _tanh(self, x):
        """Función de activación tangente hiperbólica"""
        return np.tanh(x)

    def _tanh_derivative(self, x):
        """Derivada de la función tanh"""
        return 1 - x**2

    def forward_pass(self, x):
        """
        Propagación hacia adelante.

        Args:
            x (np.ndarray): Vector de entrada

        Returns:
            tuple: (salida_capa_oculta, salida_final)
        """
        # Capa oculta
        hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self._sigmoid(hidden_input)

        # Capa de salida
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self._sigmoid(final_input)

        return hidden_output, final_output

    def backward_pass(self, x, z, hidden_output, final_output):
        """
        Propagación hacia atrás (backpropagation).

        Args:
            x (np.ndarray): Vector de entrada
            z (float): Valor esperado
            hidden_output (np.ndarray): Salida de la capa oculta
            final_output (float): Salida final de la red

        Returns:
            tuple: (gradientes de pesos y biases)
        """
        # Error en la capa de salida
        output_error = z - final_output
        output_delta = output_error * self._sigmoid_derivative(final_output)

        # Error en la capa oculta (backpropagation del error)
        hidden_error = output_delta * self.weights_hidden_output
        hidden_delta = hidden_error * self._sigmoid_derivative(hidden_output)

        # Calcular gradientes
        # Gradientes para pesos capa oculta -> salida
        grad_weights_hidden_output = output_delta * hidden_output
        grad_bias_output = output_delta

        # Gradientes para pesos entrada -> capa oculta
        grad_weights_input_hidden = np.outer(x, hidden_delta)
        grad_bias_hidden = hidden_delta

        return (grad_weights_input_hidden, grad_bias_hidden,
                grad_weights_hidden_output, grad_bias_output, output_error)

    def update_weights(self, grad_weights_input_hidden, grad_bias_hidden,
                       grad_weights_hidden_output, grad_bias_output):
        """
        Actualización de pesos y biases usando descenso de gradiente.

        Args:
            grad_weights_input_hidden: Gradientes de pesos entrada->oculta
            grad_bias_hidden: Gradientes de bias capa oculta
            grad_weights_hidden_output: Gradientes de pesos oculta->salida
            grad_bias_output: Gradiente de bias capa salida
        """
        # Actualizar pesos usando descenso de gradiente
        self.weights_input_hidden += self.learning_rate * grad_weights_input_hidden
        self.bias_hidden += self.learning_rate * grad_bias_hidden
        self.weights_hidden_output += self.learning_rate * grad_weights_hidden_output
        self.bias_output += self.learning_rate * grad_bias_output

    def train(self, X: np.ndarray, z: np.ndarray):
        """
        Entrenar la red neuronal.

        Args:
            X (np.ndarray): Conjunto de entradas
            z (np.ndarray): Conjunto de salidas esperadas (valores -1 o 1)
        """
        log.info(f"Iniciando entrenamiento: {self.epochs} épocas, lr={self.learning_rate}, epsilon={self.epsilon}")

        # Normalizar salidas de [-1, 1] a [0, 1] para sigmoid
        z_normalized = (z + 1) / 2

        convergence = False

        for epoch in tqdm(range(self.epochs), desc="Entrenando"):
            sum_squared_error = 0.0

            # Acumular gradientes para batch update
            batch_grad_w_ih = np.zeros_like(self.weights_input_hidden)
            batch_grad_b_h = np.zeros_like(self.bias_hidden)
            batch_grad_w_ho = np.zeros_like(self.weights_hidden_output)
            batch_grad_b_o = 0.0

            for x_i, z_norm_i in zip(X, z_normalized):
                # Forward pass
                hidden_output, final_output = self.forward_pass(x_i)

                # Backward pass (cálculo de error y gradientes)
                grad_w_ih, grad_b_h, grad_w_ho, grad_b_o, error = self.backward_pass(
                    x_i, z_norm_i, hidden_output, final_output
                )

                # Acumular gradientes
                batch_grad_w_ih += grad_w_ih
                batch_grad_b_h += grad_b_h
                batch_grad_w_ho += grad_w_ho
                batch_grad_b_o += grad_b_o

                # Acumular error cuadrático
                sum_squared_error += error**2

            # Actualizar pesos con gradientes acumulados
            self.update_weights(batch_grad_w_ih, batch_grad_b_h,
                              batch_grad_w_ho, batch_grad_b_o)

            # Calcular error cuadrático medio
            mean_squared_error = sum_squared_error / len(X)

            # Verificar convergencia
            if mean_squared_error < self.epsilon:
                convergence = True
                log.info(f"Convergencia alcanzada en época {epoch + 1} con MSE={mean_squared_error:.6f}")
                break

            # Logging periódico
            if (epoch + 1) % 1000 == 0:
                log.info(f"Época {epoch + 1}/{self.epochs} - MSE: {mean_squared_error:.6f}")

        final_mse = sum_squared_error / len(X)
        log.info(f"Entrenamiento finalizado después de {epoch + 1} épocas")
        log.info(f"Convergencia: {'ALCANZADA' if convergence else 'NO ALCANZADA'}")
        log.info(f"Error cuadrático medio final: {final_mse:.6f}")
        log.info(f"Pesos finales entrada->oculta:\n{self.weights_input_hidden}")
        log.info(f"Bias capa oculta: {self.bias_hidden}")
        log.info(f"Pesos finales oculta->salida: {self.weights_hidden_output}")
        log.info(f"Bias capa salida: {self.bias_output}")

    def predict(self, x: np.ndarray) -> int:
        """
        Realizar predicción con la red entrenada.

        Args:
            x (np.ndarray): Vector de entrada

        Returns:
            int: Predicción (1 o -1)
        """
        _, final_output = self.forward_pass(x)
        # Convertir a 1 o -1 (threshold en 0.5 para sigmoid)
        return 1 if final_output >= 0.5 else -1


# Función principal para probar el perceptrón
if __name__ == "__main__":
    # Datos de entrenamiento para XOR
    # Usando 1 y -1 como en los otros ejercicios
    X = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1]
    ])

    # Salidas esperadas para XOR (en formato -1, 1)
    z = np.array([-1, 1, 1, -1])

    # Crear y entrenar el perceptrón
    # Ajustado con learning_rate más alto y epsilon más bajo
    mlp = MulticapaXOR(learning_rate=1.0, epochs=10000, epsilon=0.001)
    mlp.train(X, z)

    # Probar predicciones
    log.info("\n=== PRUEBAS DE PREDICCIÓN ===")
    for x_i, z_i in zip(X, z):
        prediction = mlp.predict(x_i)
        log.info(f"Entrada: {x_i} -> Predicción: {prediction:2d}, Esperado: {z_i:2d} {'✓' if prediction == z_i else '✗'}")
