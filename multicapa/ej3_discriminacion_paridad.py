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
                 layer_one_size: int = 1, layer_two_size: int = 1, optimization_mode="descgradient"):
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
        self.alpha = 0.9

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_t = 0

        self.layer_one_size = layer_one_size
        self.layer_two_size = layer_two_size
        self.layer_two_output_size = 1
        self.num_layers = 2


        INPUT_SIZE = 7*5  # 35
        if self.optimization_mode == "momentum":
            self._initialize_delta_w(INPUT_SIZE)
        elif self.optimization_mode == "adam":
            self._initialize_adam(INPUT_SIZE)

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
        # Crea un vector simple (forma (H,)), NO una matriz de 1xH
        self.biases = [
            np.random.uniform(-0.5, 0.5, self.layer_one_size), 
            np.random.uniform(-0.5, 0.5)
        ]
        log.info(f"Red neuronal inicializada: {INPUT_SIZE} -> {self.layer_one_size} -> {self.layer_two_size} -> {self.layer_two_output_size}")

    def _initialize_delta_w(self, INPUT_SIZE):
        """
        Inicializa los delta_w previos en cero.
        Misma estructura que los pesos.
        """
        # Delta W para pesos
        self.delta_w = [
            [[0.0 for _ in range(INPUT_SIZE)] for _ in range(self.layer_one_size)],
            [0.0 for _ in range(self.layer_two_size)]
        ]
        
        # Delta B para biases
        self.delta_b = [
            np.zeros(self.layer_one_size),
            0.0
        ]
    
        log.info("Delta W inicializado para Momentum")
    
    def _initialize_adam(self, INPUT_SIZE):
        """
        Inicializa los momentos m (primer momento) y v (segundo momento) para Adam.
        Ambos se inicializan en cero.
        """
        # Primer momento (m) - equivalente a momentum
        self.m_w = [
            [[0.0 for _ in range(INPUT_SIZE)] for _ in range(self.layer_one_size)],
            [0.0 for _ in range(self.layer_two_size)]
        ]
        
        self.m_b = [
            np.zeros(self.layer_one_size),
            0.0
        ]
        
        # Segundo momento (v) - para adaptar el learning rate
        self.v_w = [
            [[0.0 for _ in range(INPUT_SIZE)] for _ in range(self.layer_one_size)],
            [0.0 for _ in range(self.layer_two_size)]
        ]
        
        self.v_b = [
            np.zeros(self.layer_one_size),
            0.0
        ]
        
        log.info("Momentos m y v inicializados para Adam")
        log.info(f"Beta1: {self.beta1}, Beta2: {self.beta2}, Epsilon: {self.adam_epsilon}")

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
        activations = [np.array(number_bit_array)]
        
        # Capa oculta
        z1 = np.dot(self.weights[0], activations[0]) + self.biases[0]
        a1 = self._sigmoid(z1)
        activations.append(a1)
        
        # Capa de salida
        z2 = np.dot(self.weights[1], a1) + self.biases[1]
        out = self._sigmoid(z2)
        activations.append(out)

        return activations

    def backward_pass(self, y_expected, activations):
        x, a1, out = activations  # x:(35,), a1:(H,), out: escalar
        y = float(y_expected)
        out = float(out)

        # Convertir a numpy arrays para operaciones vectoriales
        a1 = np.array(a1)
        x = np.array(x)

        # Error y delta de la capa de salida
        error = y - out
        delta_out = error * self._sigmoid_derivative(out)  # escalar

        # Gradientes de la capa de salida
        grad_W2 = a1 * delta_out  # Vector (H,)
        grad_b2 = delta_out      # escalar

        # Propagación a la capa oculta
        W2 = np.asarray(self.weights[1], dtype=float)  # (H,)
        
        # El delta de la capa oculta es un vector
        delta_hidden = (W2 * delta_out) * self._sigmoid_derivative(a1)  # Vector (H,)

        # Gradientes de la capa oculta
        grad_W1 = np.outer(delta_hidden, x)  # Matriz (H, 35)
        grad_b1 = delta_hidden               # Vector (H,)

        grad_weights = [grad_W1, grad_W2]
        grad_biases = [grad_b1, grad_b2]
        
        return grad_weights, grad_biases, error


    def update_weights(self, grad_weights, grad_biases):
        """
        Actualizar pesos y biases de TODAS las capas.
        """
        # Itera sobre las dos capas de pesos/biases (índices 0 y 1)
        for i in range(self.num_layers):
            # Es necesario asegurarse de que las formas de los gradientes coincidan
            # con las de los pesos. np.array ayuda a manejar esto.
            self.weights[i] += self.learning_rate * np.array(grad_weights[i])
            self.biases[i] += self.learning_rate * np.array(grad_biases[i])

    def train(self, numbers_list: np.ndarray, z) -> float:
        """
        Entrena el perceptrón usando el modo de optimización especificado.
        """

        update_functions = {
            "descgradient": self.update_weights,
            "momentum": self.update_weights_momentum,
            "adam": self.update_weights_adam
        }
        
        if self.optimization_mode not in update_functions:
            raise ValueError(f"Modo de optimización no soportado: {self.optimization_mode}")
        
        self._log_training_start()
        
        if self.optimization_mode == "adam":
            self.adam_t = 0

        z_normalized = z / 9.0
        N = len(numbers_list)
        convergence = False
        
        update_function = update_functions[self.optimization_mode]

        for epoch in tqdm.tqdm(range(self.epochs), desc=f"Entrenando ({self.optimization_mode})"):
            sum_squared_error = 0.0

            idxs = np.arange(N)
            np.random.shuffle(idxs)

            for i in idxs:
                x_i = numbers_list[i]
                y_i = z_normalized[i]
                
                activations = self.forward_pass(x_i)
                grad_w, grad_b, error = self.backward_pass(y_i, activations)
                
                update_function(grad_w, grad_b)

                sum_squared_error += float(error) ** 2

            mse = np.mean(sum_squared_error)

            if mse < self.epsilon:
                convergence = True
                log.info(f"Convergencia alcanzada en época {epoch + 1} con MSE={mse:.6f}")
                break

            if (epoch + 1) % 100 == 0:
                log.info(f"Época {epoch + 1}/{self.epochs} - MSE: {mse:.6f}")

        self._log_training_end(epoch + 1, convergence, mse)
        return mse

    def _log_training_start(self):
        """Logging del inicio del entrenamiento según el modo."""
        if self.optimization_mode == "descgradient":
            log.info(f"Iniciando entrenamiento GRADIENTE DESCENDENTE: {self.epochs} épocas, lr={self.learning_rate}")
        elif self.optimization_mode == "momentum":
            log.info(f"Iniciando entrenamiento MOMENTUM: {self.epochs} épocas, lr={self.learning_rate}, alpha={self.alpha}")
        elif self.optimization_mode == "adam":
            log.info(f"Iniciando entrenamiento ADAM: {self.epochs} épocas, lr={self.learning_rate}")
            log.info(f"Beta1={self.beta1}, Beta2={self.beta2}, Epsilon={self.adam_epsilon}")

    def _log_training_end(self, total_epochs, convergence, mse):
        """Logging del final del entrenamiento."""
        log.info(f"Entrenamiento finalizado después de {total_epochs} épocas")
        log.info(f"Convergencia: {'ALCANZADA' if convergence else 'NO ALCANZADA'}")
        log.info(f"Error cuadrático medio final: {mse:.6f}")

    def update_weights_momentum(self, grad_w, grad_b):
            """
            Momentum: Δw(t+1) = -η·∂E/∂w + α·Δw(t)
            """
            # LAYER 1
            for i in range(self.layer_one_size):
                for j in range(len(self.weights[0][i])):
                    # Δw(t+1) =η·gradiente + α·Δw(t)
                    new_delta = self.learning_rate * grad_w[0][i][j] + self.alpha * self.delta_w[0][i][j]
                    self.delta_w[0][i][j] = new_delta
                    self.weights[0][i][j] += new_delta
            
            for i in range(self.layer_one_size):
                new_delta = self.learning_rate * grad_b[0][i] + self.alpha * self.delta_b[0][i]
                self.delta_b[0][i] = new_delta
                self.biases[0][i] += new_delta
            
            # LAYER 2 
            for i in range(self.layer_two_size):
                new_delta = self.learning_rate * grad_w[1][i] + self.alpha * self.delta_w[1][i]
                self.delta_w[1][i] = new_delta
                self.weights[1][i] += new_delta
            
            new_delta = self.learning_rate * grad_b[1] + self.alpha * self.delta_b[1]
            self.delta_b[1] = new_delta
            self.biases[1] += new_delta
    
    def update_weights_adam(self, grad_w, grad_b):
        """
        Actualiza pesos usando el optimizador Adam.
        
        Adam combina momentum y RMSprop:
        - m: promedio móvil exponencial de gradientes (momentum)
        - v: promedio móvil exponencial de gradientes al cuadrado (RMSprop)
        
        Fórmulas:
            m(t) = β₁·m(t-1) + (1-β₁)·∇E
            v(t) = β₂·v(t-1) + (1-β₂)·(∇E)²
            m̂ = m / (1 - β₁ᵗ)  ← Corrección de bias
            v̂ = v / (1 - β₂ᵗ)  ← Corrección de bias
            w = w - η · m̂ / (√v̂ + ε)
        """
        # Incrementar contador de iteraciones
        self.adam_t += 1
        t = self.adam_t
        
        # Factores de corrección de bias
        bias_correction1 = 1 - (self.beta1 ** t)
        bias_correction2 = 1 - (self.beta2 ** t)
        
        # ==================== LAYER 1 - PESOS ====================
        for i in range(self.layer_one_size):
            for j in range(len(self.weights[0][i])):
                g = grad_w[0][i][j]  # Gradiente actual
                
                # Actualizar primer momento (momentum)
                self.m_w[0][i][j] = self.beta1 * self.m_w[0][i][j] + (1 - self.beta1) * g
                
                # Actualizar segundo momento (RMSprop)
                self.v_w[0][i][j] = self.beta2 * self.v_w[0][i][j] + (1 - self.beta2) * (g ** 2)
                
                # Corrección de bias
                m_hat = self.m_w[0][i][j] / bias_correction1
                v_hat = self.v_w[0][i][j] / bias_correction2
                
                # Actualizar peso
                self.weights[0][i][j] += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)
        
        # ==================== LAYER 1 - BIASES ====================
        for i in range(self.layer_one_size):
            g = grad_b[0][i]
            
            self.m_b[0][i] = self.beta1 * self.m_b[0][i] + (1 - self.beta1) * g
            self.v_b[0][i] = self.beta2 * self.v_b[0][i] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m_b[0][i] / bias_correction1
            v_hat = self.v_b[0][i] / bias_correction2
            
            self.biases[0][i] += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)
        
        # ==================== LAYER 2 - PESOS ====================
        for i in range(self.layer_two_size):
            g = grad_w[1][i]
            
            self.m_w[1][i] = self.beta1 * self.m_w[1][i] + (1 - self.beta1) * g
            self.v_w[1][i] = self.beta2 * self.v_w[1][i] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m_w[1][i] / bias_correction1
            v_hat = self.v_w[1][i] / bias_correction2
            
            self.weights[1][i] += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)
        
        # ==================== LAYER 2 - BIAS ====================
        g = grad_b[1]
        
        self.m_b[1] = self.beta1 * self.m_b[1] + (1 - self.beta1) * g
        self.v_b[1] = self.beta2 * self.v_b[1] + (1 - self.beta2) * (g ** 2)
        
        m_hat = self.m_b[1] / bias_correction1
        v_hat = self.v_b[1] / bias_correction2
        
        self.biases[1] += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)

    def predict(self, x):
        """
        Realizar predicción con la red entrenada.

        Args:
            x : array 1D - representa el numero
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
    
    def predict_parity(self, x) -> bool:
        """
        Realizar predicción de paridad con la red entrenada.
        Args:
            x : un array 1D que representa el número
        Returns:
            bool: True si el número es par, False si es impar.
        """
        activations = self.forward_pass(x)
        final_output = activations[-1]

        # Si la salida es escalar, tomar el primer elemento
        # if isinstance(final_output, np.ndarray):
        #     final_output = final_output[0] if len(final_output) == 1 else final_output

        # Convertir de [0,1] a [-1,1]
        return int(final_output * 10)%2==0