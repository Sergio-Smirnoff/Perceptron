import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from LinearPerceptron import LinearPerceptron
from NonLinearPerceptron import NonLinearPerceptron

def parse_params(params):
    with open(params) as f:
        params = json.load(f)
    learn_rates = params["learn_rates"]
    epochs = params["epochs"]
    epsilon = params["epsilon"]
    beta = params["non_linear_beta"]
    input_file = params["input_file"]
    return learn_rates, epochs, epsilon, beta, input_file

def parse_training_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    return X, y

def analisis():
    # --- Parámetros del Experimento ---
    learning_rates, epochs, epsilon, beta, input_file = parse_params("config/ej2_learn_params.json")
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

    # --- Crear figura para los plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Comparación de Error (MSE) por Época y Tasa de Aprendizaje', fontsize=16)

    # --- Experimento 1: Perceptrón Lineal ---
    print("Ejecutando experimento para Perceptrón Lineal...")
    for lr in tqdm(learning_rates, desc="Linear Perceptron"):
        perceptron = LinearPerceptron(learning_rate=lr, epochs=epochs, epsilon=epsilon)
        errors, predicts = perceptron.train(X_scaled, z_scaled)

        # Desescalar predicciones
        errors_unscaled = []
        for epoch_preds in predicts:   
            y_pred_scaled = np.array(epoch_preds)
            y_pred_real = (y_pred_scaled + 1) / 2 * range_z + min_z
            mse_real = np.mean((y_pred_real - z_train)**2)
            errors_unscaled.append(mse_real)
        ax1.plot(range(1, len(errors_unscaled) + 1), errors_unscaled, label=f'LR = {lr}')

        #ax1.plot(range(1, len(errors) + 1), errors, label=f'LR = {lr}')
        
    ax1.set_title('Perceptrón Lineal')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Error Cuadrático Medio (MSE)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(bottom=-0.02)

    # --- Experimento 2: Perceptrón No-Lineal ---
    print("\nEjecutando experimento para Perceptrón No Lineal...")
    for lr in tqdm(learning_rates, desc="Non-Linear Perceptron"):
        perceptron = NonLinearPerceptron(learning_rate=lr, epochs=epochs, beta=beta, epsilon=epsilon)
        errors, predicts = perceptron.train(X_scaled, z_scaled)

        # Desescalar predicciones
        errors_unscaled = []
        for epoch_preds in predicts:   
            y_pred_scaled = np.array(epoch_preds)
            y_pred_real = (y_pred_scaled + 1) / 2 * range_z + min_z
            mse_real = np.mean((y_pred_real - z_train)**2)
            errors_unscaled.append(mse_real)
        ax2.plot(range(1, len(errors_unscaled) + 1), errors_unscaled, label=f'LR = {lr}')

        #ax2.plot(range(1, len(errors) + 1), errors, label=f'LR = {lr}')

    ax2.set_title('Perceptrón No Lineal con Tanh')
    ax2.set_xlabel('Época')
    ax2.grid(True)
    ax2.legend()

    # --- Mostrar Plots ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots/ej2_learn.png")
    plt.show()

if __name__ == '__main__':
    analisis()