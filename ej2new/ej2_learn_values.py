import json
import pandas as pd
from LinearPerceptron import LinearPerceptron
from NonLinearPerceptron import NonLinearPerceptron

def parse_params(params):
    with open(params) as f:
        params = json.load(f)
    linear_learn_rate = params["linear_learn_rate"]
    nonlinear_learn_rate = params["nonlinear_learn_rate"]
    epochs = params["epochs"]
    epsilon = params["epsilon"]
    beta = params["non_linear_beta"]
    input_file = params["input_file"]
    return linear_learn_rate, nonlinear_learn_rate, epochs, epsilon, beta, input_file

def parse_training_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    return X, y

def main():
    # --- Parámetros del Experimento ---
    linear_learn_rate, nonlinear_learn_rate, epochs, epsilon, beta, input_file = parse_params("config/ej2_learn_values_params.json")
    X_train, z_train = parse_training_data(input_file)
    
    # --- Preprocesamiento General ---
    # Escalar X a [-1, 1] para el perceptrón no lineal
    min_x = X_train.min(axis=0)
    range_x = X_train.max(axis=0) - min_x
    range_x[range_x == 0] = 1 
    X_scaled = 2 * (X_train - min_x) / range_x - 1

    # Escalar z a [-1, 1] para AMBOS perceptrones (para comparación justa)
    min_z = z_train.min()
    range_z = z_train.max() - min_z
    range_z = 1 if range_z == 0 else range_z
    z_scaled = 2 * (z_train - min_z) / range_z - 1

    # --- Entrenamiento 1: Perceptrón Lineal ---
    linear_perceptron = LinearPerceptron(learning_rate=linear_learn_rate, epochs=epochs, epsilon=epsilon)
    linear_perceptron.train(X_scaled, z_scaled)

    # --- Entrenamiento 2: Perceptrón No-Lineal ---
    nonlinear_perceptron = NonLinearPerceptron(learning_rate=nonlinear_learn_rate, epochs=epochs, beta=beta, epsilon=epsilon)
    nonlinear_perceptron.train(X_scaled, z_scaled)

    # --- Predicciones ---
    linear_predicts = linear_perceptron.predict(X_scaled)
    nonlinear_predicts = nonlinear_perceptron.predict(X_scaled)

    # --- Desescalar predicciones a valores reales ---
    linear_unscaled = (linear_predicts + 1) / 2 * range_z + min_z
    nonlinear_unscaled = (nonlinear_predicts + 1) / 2 * range_z + min_z

    with open("results/ej2_learn_values.txt", "w") as f:
        f.write("x1,x2,x3,y_real,y_pred_linear,y_pred_nonlinear\n")
        for xi, y_real, y_lin, y_nonlin in zip(X_train, z_train, linear_unscaled, nonlinear_unscaled):
            f.write(f"{xi[0]:.4f},{xi[1]:.4f},{xi[2]:.4f},{y_real:.4f},{y_lin:.4f},{y_nonlin:.4f}\n")

if __name__ == '__main__':
    main()
