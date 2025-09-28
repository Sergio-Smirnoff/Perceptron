# SimplePerceptron.py
import numpy as np
import random
from tqdm import tqdm # Para la barra de progreso

class SimplePerceptron:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def _step_activation_function(self, x: float) -> int:
        return 1 if x >= 0 else -1
    
    def train(self, X: np.ndarray, z: np.ndarray, epochs: int = 100, epsilon: float = 1e-14):
        log_file = open("training_log.txt", "w")  
        self.weights = np.array([random.uniform(-0.5, 0.5) for _ in range(len(X[0]))])
        self.bias = random.uniform(-0.5, 0.5)
        
        for epoch in tqdm(range(epochs), desc="Training..."):
            sum_squared_error = 0.0
            
            # Acumular gradientes
            weight_gradients = np.zeros_like(self.weights)
            bias_gradient = 0.0
            
            for x_idx, x_i in enumerate(X):
                # Calculate weighted sum
                weighted_sum = np.dot(x_i, self.weights) + self.bias
                
                # Compute activation
                output = self._step_activation_function(weighted_sum)
                
                # Calculate error
                error = z[x_idx] - output
                sum_squared_error += error**2
                
                # Acumular gradientes
                weight_gradients += error * x_i
                bias_gradient += error
            
            # Actualizar parámetros una vez por época
            self.weights += self.learning_rate * weight_gradients
            self.bias += self.learning_rate * bias_gradient
            
            # Log una vez por época
            log_file.write(f"{self.weights[0]},{self.weights[1]},{self.bias}\n")
            
            # Calcular MSE correctamente
            mean_squared_error = sum_squared_error / len(X)
            convergence = mean_squared_error < epsilon
            
            if convergence: 
                break
                
        print(f"Training finished after {epoch + 1} epochs")
        print(f"Convergence was {'reached' if convergence else 'not reached'}")
        print(f"Final weights: {self.weights}")
        print(f"Final bias: {self.bias}")
        log_file.close()

    def predict(self, input:np.ndarray) -> int:
        # Calculate weighted sum
        sum = np.dot(input, self.weights) + self.bias

        # Compute activation
        output = self._step_activation_function(sum)

        return output