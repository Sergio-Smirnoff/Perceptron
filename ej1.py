# ej1
from SimplePerceptron import SimplePerceptron
import numpy as np  # ← Agregar esta importación

def main():
    # Convertir a arrays de NumPy
    training_set_and = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    expected_output_and = np.array([-1, -1, -1, 1])
    training_set_xor = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    expected_output_xor = np.array([1, 1, -1, -1])
    
    perceptron = SimplePerceptron(0.1)

    #perceptron.train(training_set_and, expected_output_and)
    perceptron.train(training_set_xor, expected_output_xor)
    
    # También convertir los inputs de prueba
    test_inputs = [
        np.array([-1, 1]),
        np.array([1, -1]),
        np.array([-1, -1]),
        np.array([1, 1])
    ]
    

    
    for input_vec in test_inputs:
        output = perceptron.predict(input_vec)
        #print(f"result for {input_vec} after training with AND was {output}")
        print(f"result for {input_vec} after training with XOR was {output}")

if __name__ == "__main__":
    main()