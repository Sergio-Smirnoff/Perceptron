import numpy as np
import random
from tqdm import tqdm

class NonLinearPerceptron():
    """
    Un perceptrón con una función de activación no lineal (tanh).
    """
    def __init__(self, learning_rate:float, epochs:int=100, epsilon:float=0.01, beta:float=1.0):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.epochs = epochs
        self.epsilon = epsilon
        self.beta = beta

    def _activation_function(self, x:float) -> float:
        """
        Función de activación: Tangente Hiperbólica.
        Devuelve valores en el rango [-1, 1].
        """
        return np.tanh(self.beta * x)

    def _activation_derivative(self, activated_output:float) -> float:
        """
        Derivada de la función de activación tanh.
        Se calcula como 1 - tanh(x)^2.
        """
        return self.beta * (1.0 - activated_output**2)

    def train(self, X: np.ndarray, z: np.array):
        """
        Entrena el perceptrón no lineal usando el descenso de gradiente.

        Args:
            X (np.ndarray): Variables de entrada del set de entrenamiento.
            z (np.array): Salidas esperadas (targets).
        """
        num_features = X.shape[1]
        self.weights = np.array([random.uniform(-0.5, 0.5) for _ in range(num_features)])
        self.bias = random.uniform(-0.5, 0.5)
        
        errors_per_epoch = []
        predicts_per_epoch = []

        for epoch in tqdm(range(self.epochs), desc="Training Non-Linear Perceptron..."):
            sum_squared_error = 0.0
            predicts = []

            for i, x_i in enumerate(X):
                # 1. Calcular la salida lineal (potencial de activación)
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # 2. Aplicar la función de activación no lineal
                output = self._activation_function(linear_output)
                predicts.append(output)

                # 3. Calcular el error
                error = z[i] - output
                sum_squared_error += error**2

                # 4. Calcular la derivada de la función de activación
                derivative = self._activation_derivative(output)
                
                # 5. Calcular el delta para la actualización
                delta = self.learning_rate * error * derivative

                # 6. Actualizar pesos y bias
                self.weights += delta * x_i
                self.bias += delta

            mean_squared_error = sum_squared_error / len(X)
            errors_per_epoch.append(mean_squared_error)
            predicts_per_epoch.append(predicts)
            
            # Condición de parada por error mínimo
            if mean_squared_error < self.epsilon:
                print(f"\nConvergencia alcanzada en la época {epoch+1} con un MSE de {mean_squared_error:.6f}")
                break
        
        print(f"\nEntrenamiento finalizado.")
        print(f"Pesos finales: {self.weights}")
        print(f"Bias final: {self.bias}")
        
        return errors_per_epoch, predicts_per_epoch

    def train_and_test(self, X: np.ndarray, z: np.array, X_test: np.ndarray, z_test: np.array):
        """
        Entrena el perceptrón no lineal usando el descenso de gradiente.
        Testea en cada epoca.

        Args:
            X (np.ndarray): Variables de entrada del set de entrenamiento.
            z (np.array): Salidas esperadas (targets).
        """
        num_features = X.shape[1]
        self.weights = np.array([random.uniform(-0.5, 0.5) for _ in range(num_features)])
        self.bias = random.uniform(-0.5, 0.5)
        
        errors_per_epoch = []
        predicts_per_epoch = []
        test_errors_per_epoch = []
        test_predicts_per_epoch = []

        for epoch in tqdm(range(self.epochs), desc="Training Non-Linear Perceptron..."):
            sum_squared_error = 0.0
            predicts = []

            for i, x_i in enumerate(X):
                # 1. Calcular la salida lineal (potencial de activación)
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # 2. Aplicar la función de activación no lineal
                output = self._activation_function(linear_output)
                predicts.append(output)

                # 3. Calcular el error
                error = z[i] - output
                sum_squared_error += error**2

                # 4. Calcular la derivada de la función de activación
                derivative = self._activation_derivative(output)
                
                # 5. Calcular el delta para la actualización
                delta = self.learning_rate * error * derivative

                # 6. Actualizar pesos y bias
                self.weights += delta * x_i
                self.bias += delta

            mean_squared_error = sum_squared_error / len(X)
            errors_per_epoch.append(mean_squared_error)
            predicts_per_epoch.append(predicts)

            # Test

            # 1. Calcular predicciones
            test_outputs = self.predict(X_test)
            test_predicts_per_epoch.append(test_outputs)

            # 2. Calcular el error
            test_errors = z_test - test_outputs
            test_mean_squared_error = np.sum(test_errors**2) / len(X_test)
            test_errors_per_epoch.append(test_mean_squared_error)
            
            # Condición de parada por error mínimo
            if mean_squared_error < self.epsilon:
                print(f"\nConvergencia alcanzada en la época {epoch+1} con un MSE de {mean_squared_error:.6f}")
                break
        
        print(f"\nEntrenamiento finalizado.")
        print(f"Pesos finales: {self.weights}")
        print(f"Bias final: {self.bias}")
        
        return errors_per_epoch, predicts_per_epoch, test_errors_per_epoch, test_predicts_per_epoch


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la salida para un vector de entrada dado.
        
        Args:
            X (np.ndarray): Vector de entrada.
            
        Returns:
            np.ndarray: Salida predicha después de aplicar la función de activación.
        """ 
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation_function(linear_output)
