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
            [[random.uniform(-0.5, 0.5) for _ in range(self.layer_one_size)] for _ in range(INPUT_SIZE)],  # 35 inputs to layer one
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

    def forward_pass(self, numbers_list):
        """
        Propagación hacia adelante (matricial).
        numbers_list: vector 1D de tamaño 35 (bits del número)
        """
        activations = [numbers_list]

        # Capa oculta
        z1 = np.dot(numbers_list, np.array(self.weights[0])) + np.array(self.biases[0]).flatten()
        a1 = self._sigmoid(z1)
        activations.append(a1)

        # Capa de salida
        z2 = np.dot(a1, np.array(self.weights[1])) + self.biases[1]
        a2 = self._sigmoid(z2)
        activations.append(a2)

        return activations

    def backward_pass(self, z_expected, activations):
        """
        Propagación hacia atrás (backpropagation) por TODAS las capas.
        Usa regla de la cadena para calcular gradientes.

        Args:
            z_expected (float): Valor esperado
                z = [expected_l1(), expected_l2()]

            activations (list): Salidas de cada capa (del forward pass)
                    act = [input_original, [activation_l11(), activation_l12(),....], activation_l2()]

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

        expected_l2 = z_expected #TODO check
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

    def train(self, bit_matrix: np.ndarray, z):
        """
        Entrenar la red neuronal multicapa.

        Args:
            bit_matrix (np.ndarray): Conjunto de entradas
            z (np.ndarray): Conjunto de salidas esperadas
        """
        if self.optimization_mode == "descgradient":
            self.train_desceding_gradient(bit_matrix, z)
        else:
            raise NotImplementedError(f"Método de optimización '{self.optimization_mode}' no implementado.")

    def train_desceding_gradient(self, numbers_list: np.ndarray, z):
        """
        Entrenar la red neuronal multicapa.

        Args:
            numbers_list (np.ndarray): matrices de numeros
            nl = [ [bits_de_1], [bits_de_2], ... ]
            z (np.ndarray): Conjunto de salidas esperadas
        """
        log.info(f"Iniciando entrenamiento: {self.epochs} épocas, lr={self.learning_rate}")

        # Normalizar salidas de [-1, 1] a [0, 1] para sigmoid
        # z_normalized = (z + 1) / 2
        z_normalized = z/9.0 #TODO check -> normalize by biggest expected value = 9

        convergence = False

        for epoch in tqdm.tqdm(range(self.epochs), desc="Entrenando"):
            sum_squared_error = 0.0

            # Acumular gradientes para batch update
            batch_grad_w = [np.zeros_like(w) for w in self.weights]
            batch_grad_b = [np.zeros_like(b) for b in self.biases]


            for x_i, z_i in zip(numbers_list, z_normalized):
                #cada x_i es un numero en forma array 1D de bits
                # z_i son los numeros esperados en la salida

                activations = self.forward_pass(x_i)

                # 2. Backward pass
                grad_w, grad_b, error = self.backward_pass(z_i, activations)

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
            mean_squared_error = sum_squared_error / len(numbers_list)

            # Verificar convergencia
            if mean_squared_error < self.epsilon:
                convergence = True
                log.info(f"Convergencia alcanzada en época {epoch + 1} con MSE={mean_squared_error:.6f}")
                break

            if (epoch + 1) % 1000 == 0:
                log.info(f"Época {epoch + 1}/{self.epochs} - MSE: {mean_squared_error:.6f}")

        final_mse = sum_squared_error / len(numbers_list)
        log.info(f"Entrenamiento finalizado después de {epoch + 1} épocas")
        log.info(f"Convergencia: {'ALCANZADA' if convergence else 'NO ALCANZADA'}")
        log.info(f"Error cuadrático medio final: {final_mse:.6f}")

    def predict(self, x):
        """
        Realizar predicción con la red entrenada.

        Args:
            x (np.ndarray): Vector de entrada -> es la matriz de bits pasada a array 1D
            x = [f1, f2, ..., f7]

        Returns:
            int: Predicción (1 o -1)
        """
        activations = self.forward_pass(x)
        final_output = activations[-1]

        # Si la salida es escalar, tomar el primer elemento
        # if isinstance(final_output, np.ndarray):
        #     final_output = final_output[0] if len(final_output) == 1 else final_output

        # Convertir de [0,1] a [-1,1]
        return int(final_output * 10)