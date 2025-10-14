import sys
import logging as log
import numpy as np

class MultiLayer:
    
    def __init__(
            self, 
            layers_array, # array que contiene la cantidad de neuronas por capa, el size es la cantidad de capas 
            learning_rate=0.01, 
            epochs=1000, 
            epsilon=1e-3, 
            optimization_mode="descgradient",
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
        delta = np.array([])
        
        # ====== ÚLTIMA CAPA - DELTA SEGÚN FUNCIÓN DE PÉRDIDA ======
        if self.loss_function == "cross_entropy":
            delta = activations[-1] - y_one_hot
        
        elif self.loss_function == "mse":
            error = activations[-1] - y_one_hot
            delta = error * self._sigmoid_derivative(activations[-1])
        
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
    def update_momentum(self, weights_gradients, biases_gradients, beta=0.9):
        """Actualización con momentum""" 
        pass

    # Train y predict
    def train(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        for epoch in range(self.epochs):
            self.log.debug(f"Epoch {epoch+1}/{self.epochs}")
            
            activations = self.forward_pass(X_train)
            
            loss_entropy = self.compute_loss(activations[-1], y_train)
            loss_mse = self.compute_mse(activations[-1], y_train)
            
            weights_gradients, biases_gradients = self.backward_pass(activations, y_train)
            
            # ====== ACTUALIZAR PARÁMETROS ======
            if self.optimization_mode == "edg":
                self.update_desc_gradient(weights_gradients, biases_gradients)
            elif self.optimization_mode == "momentum":
                self.update_momentum(weights_gradients, biases_gradients)
            
            # ====== VERIFICAR CONVERGENCIA con la función de pérdida elegida ======
            if self.loss_function == "cross_entropy":
                current_loss = loss_entropy
                prev_loss = self.error_entropy_ant
            else:  # mse
                current_loss = loss_mse
                prev_loss = self.error_mse_ant
            
            self.error_entropy = loss_entropy
            self.error_mse = loss_mse

            # Guardar el mejor error
            if loss_entropy < self.error_entropy_min:
                self.error_entropy_min = loss_entropy

            if loss_mse < self.error_mse_min:
                self.error_mse_min = loss_mse
            
            # Criterio de parada
            if abs(prev_loss - current_loss) < self.epsilon:
                self.log.debug(f"Convergencia alcanzada en época {epoch}")
                break

            self.error_mse_ant = loss_mse
            self.error_entropy_ant = loss_entropy

        self.log.debug(f"Entrenamiento completado. Error mínimo: {self.error_entropy_min:.4f}, MSE: {self.error_mse_min:.4f}")
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

    def predict_proba(self, X):
        """
        Retorna las probabilidades para cada clase
        X: (batch_size, n_features)
        Retorna: (batch_size, 10) con probabilidades
        """
        activations = self.forward_pass(X)
        return activations[-1]

# Codigos viejos antes de refactorizar para aceptar mse y cross-entropy
"""
# Train y predict
    def train(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

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
            # elif self.optimization_mode == "adam":
            #     self.update_adam(weights_gradients, biases_gradients)
            
            # ====== VERIFICAR CONVERGENCIA ======
            self.error_entropy = loss
            self.error_mse = mse_loss

            # Guardar el mejor error
            if loss < self.error_entropy_min:
                self.error_entropy_min = loss

            if mse_loss < self.error_mse_min:
                self.error_mse_min = mse_loss
            
            # Criterio de parada: mejora menor que epsilon
            if abs(self.error_entropy_ant - loss) < self.epsilon:
                self.log.debug(f"Convergencia alcanzada en época {epoch}")
                break

            self.error_mse_ant = self.error_mse
            self.error_entropy_ant = self.error_entropy

        self.log.debug(f"Entrenamiento completado. Error mínimo: {self.error_entropy_min:.4f}")
        # Ver si devolver el error minimo
        return self.error_entropy, self.error_mse

"""
"""
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

"""