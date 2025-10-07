import random
import numpy as np
import logging as log
import tqdm


log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ParityMultyPerceptron:
    """
    Perceptrón multicapa para aprender la función de paridad.
    Arquitectura: 2 entradas -> capa oculta (3 neuronas) -> 1 salida
    """

    def __init__(self, learning_rate: float, epochs: int = 100, epsilon: float = 0.0, 
                 layer_one_size: int = 1, layer_two_size: int = 10, optimization_mode="descgradient"):
        """
        Args:
            learning_rate (float): Tasa de aprendizaje
            epochs (int): Número de épocas de entrenamiento
            epsilon (float): Error mínimo para convergencia

            optimization modes: {descgradient, momentum, adam}
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.optimization_mode = optimization_mode

        # Arquitectura de la red: 2 entradas, 3 neuronas ocultas, 1 salida
        self.layer_one_size = layer_one_size
        self.layer_two_size = layer_two_size
        self.layer_two_output_size = 1
        self.num_layers = 2


        INPUT_SIZE = 7*5  # 35
        #hardcoded for 2 layers
        # weights = [ 
        #       [ layer 1
        #           [w1, w2, ..., w35], ..., [w1, w2, ..., w35]  # for each neuron in layer one
        #       ], 
        #       [ layer 2
        #           [p1, p2, ..., p10]
        #       ]
        # ]
        self.weights = [ 
            [[random.uniform(-0.5, 0.5) for _ in range(INPUT_SIZE)] for _ in range(self.layer_one_size)],  # 35 inputs to layer one
            [random.uniform(-0.5, 0.5) for _ in range(self.layer_two_size)]]


        # biases = [
        #           [b1, b2, ..., b35],  # for layer one
        #           [B1, B2, ..., B10]   # for layer two
        # ]
        self.biases = [ [[random.uniform(-0.5, 0.5) for _ in range(self.layer_one_size)]], random.uniform(-0.5, 0.5)]

        log.info(f"Red neuronal inicializada: {INPUT_SIZE} -> {self.layer_one_size} -> {self.layer_two_size} -> {self.layer_two_output_size}")

    def _sigmoid(self, x):
        """Función de activación sigmoide
        Rango: (0, 1)
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip para evitar overflow

    def _sigmoid_derivative(self, x):
        """Derivada de la función sigmoide"""
        return x * (1 - x)

    def _tanh(self, x):
        """Función de activación tangente hiperbólica
        Rango: (-1, 1)
        """
        return np.tanh(x)

    def _tanh_derivative(self, x):
        """Derivada de la función tanh"""
        return 1 - x**2

    def forward_pass(self, number_bit_array):
        """
        Propagación hacia adelante por TODAS las capas.

        Args:
            number_bit_array: array 1D - representa los numeros
                input = [f1, f2, f3, f4, f5, f6, f7]  (7x5)
            se asume que ya esta aplanada a array 1D

        Returns:
            list: Lista con salidas de cada capa [h1, h2, ..., hN, o]
            las capas que se componen de multiples neuronas son listas
            act = [[activation_l11(), activation_l12(),....], activation_l2()]
        """

        # activations = [ entrada matriz = [7x5] , output-one = [7x1],output-two = [1x1]]
        #act = [f1, f2, f3]
        activations = [number_bit_array]  # Guardar activación de cada capa (empezando por entrada)
        #activations = [x, layer1, layer2]

        matrix = []
        current_input = number_bit_array
        for j in range(self.layer_one_size):
            # z = W * a + b
            z = np.dot(np.array(current_input), self.weights[0][j]) + self.biases[0][j]
            # a = σ(z)
            activation = self._sigmoid(z)
            matrix.append(activation)

        activations.append(matrix)
        current_input = matrix

        # Capa de salida
        z = np.dot(current_input, self.weights[1]) + self.biases[1]
        activations.append(self._sigmoid(z))

        return activations

    def backward_pass(self, y_expected, activations):
        """
        y_expected: escalar (por ej. 0/1 para paridad o tu valor normalizado)
        activations: [x, a1, a2] como devuelve forward_pass
        """
        x, a1, out = activations                 # x:(35,), a1:(H,), out: escalar
        y = float(y_expected)
        out = float(out)

        # error y delta de salida
        error = y - out
        delta_out = error * self._sigmoid_derivative(out)        # escalar

        # Gradientes capa de salida (formas = pesos reales)
        grad_W2 = a1[0] * delta_out                                 # (H,)
        grad_b2 = delta_out                                      # escalar

        # Propagación a capa oculta
        W2 = np.asarray(self.weights[1], dtype=float)            # (H,)
        delta_hidden = (W2 * delta_out) * self._sigmoid_derivative(a1[0])  # (H,)

        # Gradientes capa oculta
        grad_W1 = np.outer(delta_hidden, x)                      # (H, 35)
        grad_b1 = delta_hidden                                   # (H,)

        grad_weights = [grad_W1, grad_W2]
        grad_biases  = [grad_b1, grad_b2]
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

    def train(self, numbers_list: np.ndarray, z: np.ndarray):
        """
        Entrenar la red neuronal multicapa.

        Args:
            numbers_list (np.ndarray): Conjunto de entradas
            z (np.ndarray): Conjunto de salidas esperadas
        """
        if self.optimization_mode == "descgradient":
            self.train_desceding_gradient(numbers_list, z)
        else:
            raise NotImplementedError(f"Método de optimización '{self.optimization_mode}' no implementado.")

    # def train_desceding_gradient(self, numbers_list: np.ndarray, z):
    #     """
    #     Entrenar la red neuronal multicapa.

    #     Args:
    #         numbers_list (np.ndarray): matrices de numeros
    #         nl = [ [bits_de_1], [bits_de_2], ... ]
    #         z (np.ndarray): Conjunto de salidas esperadas
    #     """
    #     log.info(f"Iniciando entrenamiento: {self.epochs} épocas, lr={self.learning_rate}")

    #     # Normalizar salidas de [-1, 1] a [0, 1] para sigmoid
    #     # z_normalized = (z + 1) / 2
    #     z_normalized = z/9.0 #TODO check -> normalize by biggest expected value = 9
        
    #     convergence = False

    #     for epoch in tqdm.tqdm(range(self.epochs), desc="Entrenando"):
    #         sum_squared_error = 0.0

    #         # Acumular gradientes para batch update
    #         batch_grad_w = [np.zeros_like(w) for w in self.weights]
    #         batch_grad_b = [np.zeros_like(b) for b in self.biases]

    #         activations = self.forward_pass(numbers_list)

    #         for x_i, z_i in zip(numbers_list, z_normalized):
    #             #cada x_i es un numero en forma array 1D de bits
    #             # z_i es el numero correspondiente - valor esperado

    #             # 2. Backward pass
    #             grad_w, grad_b, error = self.backward_pass(z_i, activations)

    #             # 3. Acumular gradientes
    #             for i in range(self.num_layers - 1):
    #                 batch_grad_w[i] += grad_w[i]
    #                 batch_grad_b[i] += grad_b[i]

    #             # Asegurar que error sea escalar para la suma
    #             error_scalar = float(error) if isinstance(error, (np.ndarray, np.generic)) else error
    #             sum_squared_error += error_scalar**2

    #         # 4. Actualizar pesos con gradientes acumulados
    #         self.update_weights(batch_grad_w, batch_grad_b)

    #         # Calcular MSE
    #         mean_squared_error = sum_squared_error / len(numbers_list)

    #         # Verificar convergencia
    #         if mean_squared_error < self.epsilon:
    #             convergence = True
    #             log.info(f"Convergencia alcanzada en época {epoch + 1} con MSE={mean_squared_error:.6f}")
    #             break

    #         if (epoch + 1) % 1000 == 0:
    #             log.info(f"Época {epoch + 1}/{self.epochs} - MSE: {mean_squared_error:.6f}")

    #     final_mse = sum_squared_error / len(numbers_list)
    #     log.info(f"Entrenamiento finalizado después de {epoch + 1} épocas")
    #     log.info(f"Convergencia: {'ALCANZADA' if convergence else 'NO ALCANZADA'}")
    #     log.info(f"Error cuadrático medio final: {final_mse:.6f}")

    def train_desceding_gradient(self, numbers_list: np.ndarray, z):
        log.info(f"Iniciando entrenamiento: {self.epochs} épocas, lr={self.learning_rate}")

        # Si tu tarea es paridad 0/1, NO normalices por 9.0. Dejalo como y∈{0,1}.
        # Si estás regresando 0..9 con una sola neurona (no recomendado), entonces:
        z_normalized = z / 9.0

        N = len(numbers_list)
        convergence = False

        for epoch in tqdm.tqdm(range(self.epochs), desc="Entrenando"):
            sum_squared_error = 0.0

            # Mezcla de índices para SGD
            idxs = np.arange(N)
            np.random.shuffle(idxs)

            for i in idxs:
                x_i = numbers_list[i]   # xi es un numero en forma array 1D de bits
                y_i = z_normalized[i]   # para paridad usar y_i = z[i]

                # Forward por muestra
                activations = self.forward_pass(x_i)

                # Backward por muestra
                grad_w, grad_b, error = self.backward_pass(y_i, activations)

                # Update INMEDIATO (SGD, sin acumulación)
                self.update_weights(grad_w, grad_b)

                sum_squared_error += float(error) ** 2

            mse = sum_squared_error / N

            if mse < self.epsilon:
                convergence = True
                log.info(f"Convergencia alcanzada en época {epoch + 1} con MSE={mse:.6f}")
                break

            if (epoch + 1) % 100 == 0:
                log.info(f"Época {epoch + 1}/{self.epochs} - MSE: {mse:.6f}")

        log.info(f"Entrenamiento finalizado después de {epoch + 1} épocas")
        log.info(f"Convergencia: {'ALCANZADA' if convergence else 'NO ALCANZADA'}")
        log.info(f"Error cuadrático medio final: {mse:.6f}")

    def predict(self, x):
        """
        Realizar predicción con la red entrenada.

        Args:
            x : lista de arrays 1D - representan los numeros
            x = [array_0, array_1, ..., array_9]  (7x5 cada uno)

        Returns:
            int: Digito predicho {0, 1, ..., 9}
        """
        activations = self.forward_pass(x)
        final_output = activations[-1]

        # Si la salida es escalar, tomar el primer elemento
        # if isinstance(final_output, np.ndarray):
        #     final_output = final_output[0] if len(final_output) == 1 else final_output

        # Convertir de [0,1] a [-1,1]
        return int(final_output * 10)