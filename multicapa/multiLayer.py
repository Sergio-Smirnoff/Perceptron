import sys
import logging as log
import numpy as np

class MultiLayer:
    
    def __init__(
            self, 
            layers_array, # array que contiene la cantidad de neuronas por capa, el size es la cantidad de capas 
            learning_rate=0.01, 
            epochs=1000, 
            epsilon=1e-4, 
            optimization_mode="adam",
            loss_function="cross_entropy", # "cross_entropy" o "mse"
            seed=42):
        """
        layers: list of integers indicating the number of neurons in each layer
        learning_rate: learning rate for weight updates
        epochs: maximum number of training epochs
        epsilon: threshold for stopping criterion based on MSE improvement
        optimization_mode: "descgradient", "momentum", or "adam"
        """
        self.layers = layers_array
        self.learning_rate = learning_rate
        self.log = log.getLogger('MultiLayer')

        # cotas
        self.epochs = epochs
        self.epsilon = epsilon
        self.optimization_mode = optimization_mode
        self.seed = seed
        
        # Errores
        self.error_entropy = 1
        self.error_entropy_ant = np.inf
        self.error_entropy_min = np.inf
        self.loss_function = loss_function  # "cross_entropy" o "mse"

        self.error_mse = 1
        self.error_mse_ant = np.inf
        self.error_mse_min = np.inf

        self.weights = []  # weights for each layer
        self.biases = []   # biases for each layer
        
        self._initialize_weights()

    def _initialize_weights(self):
        import numpy as np
        np.random.seed(self.seed)  # for reproducibility
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            # Inicializacion Xavier para sigmoid
            weight_matrix = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1.0 / self.layers[i])
            # Inicializacion de bias en 0
            bias_vector = np.zeros((1, self.layers[i+1]))
            # Guardar pesos y bias
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

            self.log.debug(f"layer {i} weights shape: {weight_matrix.shape}")
            self.log.debug(f"layer {i} biases shape: {bias_vector.shape}")

        if self.optimization_mode == "adam":
            # Inicializar momentos para Adam
            self.m_weights = [np.zeros_like(w) for w in self.weights] # momentum
            self.v_weights = [np.zeros_like(w) for w in self.weights] # velocity
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.adam_epsilon = 1e-8
            self.t = 0  # contador de iteraciones para Adam
        elif self.optimization_mode == "momentum":
            # Inicializar velocidades para Momentum
            self.delta_weights = [np.zeros_like(w) for w in self.weights]
            self.delta_biases = [np.zeros_like(b) for b in self.biases]
            self.alpha = 0.9  # coeficiente de momentum (típicamente 0.9)

    ## Funciones de activación y sus derivadas

    # Sigmoid

    def _sigmoid(self, x):
        """Función de activación sigmoide
        Rango: (0, 1)
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip para evitar overflow

    def _sigmoid_derivative(self, x):
        """Derivada de la función sigmoide"""
        return x * (1 - x)
    
    # Softmax
    def _softmax(self, x):
        """Función softmax estable numéricamente"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # Cross-entropy loss
    def compute_loss(self, predictions, y_true):
        """
        Cross-Entropy Loss
        predictions: (batch_size, 10) - probabilidades de softmax
        y_true: (batch_size,) - labels como enteros 0-9
        """
        batch_size = predictions.shape[0]
        
        # Convertir y_true a índices si es necesario
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Cross-entropy usando indexing avanzado
        # Evitar log(0) con clip
        predictions_clipped = np.clip(predictions, self.epsilon, 1.0)
        
        # Seleccionar solo las probabilidades de las clases correctas
        correct_confidences = predictions_clipped[np.arange(batch_size), y_true]
        # Cross-entropy loss
        loss = -np.mean(np.log(correct_confidences))
        
        return loss
    
    # Mean Squared Error (MSE)
    def compute_mse(self, predictions, y_true):
        """
        Mean Squared Error
        predictions: (batch_size, 10) - salida de softmax
        y_true: (batch_size,) - labels como enteros 0-9
        """
        batch_size = predictions.shape[0]
        
        if len(y_true.shape) == 1:
            y_one_hot = self.to_one_hot(y_true, num_classes=self.layers[-1])
        else:
            y_one_hot = y_true
        
        mse = 0.5 * np.mean((predictions - y_one_hot) ** 2)
        
        return mse

    # Funciones basicas de la red

    def to_one_hot(self, y, num_classes=10):
        """
        y: array de enteros, shape (batch_size,)
        num_classes: número de clases (10 en tu caso)
        """
        batch_size = len(y)
        y_one_hot = np.zeros((batch_size, num_classes))
        y_one_hot[np.arange(batch_size), y] = 1
        return y_one_hot

    def forward_pass(self, X):
        """
        X: múltiples trainings => batch
        Retorna: predicciones para todos los ejemplos
        """
        activations = [X]  # Lista para guardar activaciones de cada capa
        activation = X
        for i in range(len(self.weights)-1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self._sigmoid(z)
            activations.append(activation)

        # Capa de salida con softmax
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        activation = self._softmax(z)
        activations.append(activation)

        return activations
    
    def backward_pass(self, activations, y_true):
        batch_size = activations[0].shape[0]
        
        if len(y_true.shape) == 1:
            y_one_hot = self.to_one_hot(y_true, num_classes=self.layers[-1])
        else:
            y_one_hot = y_true
        
        weights_gradients = []
        biases_gradients = []
        
        # ====== ÚLTIMA CAPA (softmax + cross-entropy) ======
        delta = activations[-1] - y_one_hot
        
        # Calcular gradientes de la última capa
        w_grad = np.dot(activations[-2].T, delta) / batch_size
        b_grad = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        weights_gradients.insert(0, w_grad)
        biases_gradients.insert(0, b_grad)
        
        self.log.debug(f"Last layer delta shape: {delta.shape}")
        
        # ====== CAPAS OCULTAS (sigmoid) ======
        for i in range(len(self.weights) - 2, -1, -1):
            # Propagar el error hacia atrás
            delta = np.dot(delta, self.weights[i + 1].T)
            
            # Aplicar derivada de sigmoid
            delta = delta * self._sigmoid_derivative(activations[i + 1])
            
            # Calcular gradientes para esta capa
            w_grad = np.dot(activations[i].T, delta) / batch_size
            b_grad = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            weights_gradients.insert(0, w_grad)
            biases_gradients.insert(0, b_grad)

            self.log.debug(f"Layer {i} delta shape: {delta.shape}")

        return weights_gradients, biases_gradients


    # Gradiente descendente
    # Actualizacion de pesos y biases
    def update_desc_gradient(self, weights_gradients, biases_gradients):
        """Actualización vectorizada de pesos y biases"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weights_gradients[i]
            self.biases[i] -= self.learning_rate * biases_gradients[i]
        self.log.debug("Weights and biases updated.")

    # Momentum
    def update_momentum(self, gradients_w, gradients_b):
        """
        Actualización de parámetros usando Momentum
        Según ecuación 8.12: Δw(t+1) = -η·∂E/∂w + α·Δw(t)
        """
        for i in range(len(self.weights)):

            delta_w_new = -self.learning_rate * gradients_w[i] + self.alpha * self.delta_weights[i]
            self.weights[i] = self.weights[i] + delta_w_new
            self.delta_weights[i] = delta_w_new
            
            delta_b_new = -self.learning_rate * gradients_b[i] + self.alpha * self.delta_biases[i]
            self.biases[i] = self.biases[i] + delta_b_new
            self.delta_biases[i] = delta_b_new

    # Adam
    def update_adam(self, gradients_w, gradients_b):
        """
        Actualización de parámetros usando Adam optimizer
        """
        # Incrementar contador de iteraciones
        self.t += 1
        
        for i in range(len(self.weights)):

            gt_w = gradients_w[i]
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * gt_w
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (gt_w ** 2)

            m_w_corrected = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_w_corrected = self.v_weights[i] / (1 - self.beta2 ** self.t)
            self.weights[i] = self.weights[i] - self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.adam_epsilon)
            gt_b = gradients_b[i]
            
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * gt_b
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (gt_b ** 2)
            
            m_b_corrected = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_b_corrected = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            self.biases[i] = self.biases[i] - self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.adam_epsilon)

    # Train y predict
    def train(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if self.error_entropy_min < self.epsilon:
            self.log.debug("El entrenamiento ya ha convergido previamente.")
            return self.error_entropy_min, self.error_mse_min
        for epoch in range(self.epochs):
            self.log.debug(f"Epoch {epoch+1}/{self.epochs}")
            # ====== FORWARD PASS ======
            activations = self.forward_pass(X_train)
            
            # ====== CALCULAR PÉRDIDA ======
            loss = self.compute_loss(activations[-1], y_train)
            mse_loss = self.compute_mse(activations[-1], y_train)
            
            # ====== BACKWARD PASS ======
            weights_gradients, biases_gradients = self.backward_pass(activations, y_train)
            
            # ====== ACTUALIZAR PARÁMETROS ======
            if self.optimization_mode == "edg":
                self.update_desc_gradient(weights_gradients, biases_gradients)
            elif self.optimization_mode == "momentum":
                self.update_momentum(weights_gradients, biases_gradients)
            elif self.optimization_mode == "adam":
                self.update_adam(weights_gradients, biases_gradients)
            
            # ====== VERIFICAR CONVERGENCIA ======
            self.error_entropy = loss
            self.error_mse = mse_loss

            # Guardar el mejor error
            if loss < self.error_entropy_min:
                self.error_entropy_min = loss

            if mse_loss < self.error_mse_min:
                self.error_mse_min = mse_loss
            
            # Criterio de parada: mejora menor que epsilon
            self.log.debug(f"Epoch {epoch+1}: Cross-Entropy Loss = {self.error_entropy:.6f},Epsilon = {self.epsilon}, MSE = {self.error_mse:.6f}")
            if self.error_entropy_min < self.epsilon:
                self.log.debug(f"Convergencia alcanzada en época {epoch}")
                break

            self.error_mse_ant = self.error_mse
            self.error_entropy_ant = self.error_entropy

        self.log.debug(f"Entrenamiento completado. Error mínimo: {self.error_entropy_min:.4f}")
        # Ver si devolver el error minimo
        return self.error_entropy, self.error_mse

    def predict(self, X):
        """
        Retorna las clases predichas (0-9)
        X: (batch_size, n_features)
        Retorna: (batch_size,) con valores 0-9
        """
        activations = self.forward_pass(X)
        predictions = activations[-1]
        
        return np.argmax(predictions, axis=1)

    # debug
    def predict_proba(self, X):
        """
        Retorna las probabilidades para cada clase
        X: (batch_size, n_features)
        Retorna: (batch_size, 10) con probabilidades
        """
        activations = self.forward_pass(X)
        return activations[-1]
