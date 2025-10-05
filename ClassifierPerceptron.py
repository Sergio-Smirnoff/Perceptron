import numpy as np
import random
from tqdm import tqdm

# 1. La función de activación ahora es una función escalón (step function).
def step_function(x: float) -> int:
    """Devuelve 1 si la entrada es >= 0, de lo contrario 0."""
    return 1 if x >= 0 else 0

class ClassifierPerceptron:
    def __init__(self, learning_rate: float, epochs: int = 100):
        # El epsilon y el threshold ya no son necesarios en este algoritmo.
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X: np.ndarray, y: np.array):
        """Entrena el perceptrón usando la regla de aprendizaje de clasificación."""
        num_features = X.shape[1]
        self.weights = np.random.uniform(-0.5, 0.5, num_features)
        self.bias = random.uniform(-0.5, 0.5)

        for epoch in range(self.epochs):
            errors_in_epoch = 0
            for i, x_i in enumerate(X):
                # Calcular la salida lineal (potencial de activación)
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # 2. Obtener la predicción APLICANDO LA FUNCIÓN ESCALÓN
                y_predicted = step_function(linear_output)
                
                # 3. Calcular el error de CLASIFICACIÓN (será -1, 0, o 1)
                error = y[i] - y_predicted
                
                # Solo actualizamos los pesos si hubo un error
                if error != 0:
                    errors_in_epoch += 1
                    update = self.learning_rate * error
                    self.weights += update * x_i
                    self.bias += update
            
            # Si no hubo errores en toda una época, el modelo ha convergido
            if errors_in_epoch == 0:
                # print(f"Convergencia alcanzada en la época {epoch + 1}") # Opcional
                return

    def predict(self, X: np.ndarray) -> np.array:
        """Predice la clase para un conjunto de entradas usando la función escalón."""
        linear_output = np.dot(X, self.weights) + self.bias
        
        # Vectorizamos la función para que se aplique a cada elemento si X es una matriz
        vectorized_step = np.vectorize(step_function)
        return vectorized_step(linear_output)