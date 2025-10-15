import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from NonLinearPerceptron import NonLinearPerceptron

def parse_params(params):
    with open(params) as f:
        params = json.load(f)
    nonlinear_learn_rate = params["nonlinear_learn_rate"]
    epochs = params["epochs"]
    epsilon = params["epsilon"]
    beta = params["non_linear_beta"]
    input_file = params["input_file"]
    k_values = params["k_values"]
    return nonlinear_learn_rate, epochs, epsilon, beta, input_file, k_values

def parse_training_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    return X, y

def k_fold_indices(n_samples, k, seed=0):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)
    return folds

def main():
    # --- Parámetros del Experimento ---
    nonlinear_learn_rate, epochs, epsilon, beta, input_file, k_values = parse_params("config/ej2_generalize_params.json")
    X_values, z_values = parse_training_data(input_file)

    # Variables para el plot general
    all_k_results = {}
    colors = plt.cm.get_cmap('viridis', len(k_values)) 

    # --- Entrenamiento por K valor ---
    for idx, k in enumerate(k_values):
        train_curves = []
        test_curves = []

        kf_idx = k_fold_indices(len(X_values), k, seed=42)

        # --- Preprocesamiento General ---
        # Escalar X a [-1, 1] para el perceptrón no lineal
        min_x = X_values.min(axis=0)
        range_x = X_values.max(axis=0) - min_x
        range_x[range_x == 0] = 1 
        X_scaled = 2 * (X_values - min_x) / range_x - 1

        # Escalar z a [-1, 1]
        min_z = z_values.min()
        range_z = z_values.max() - min_z
        range_z = 1 if range_z == 0 else range_z
        z_scaled = 2 * (z_values - min_z) / range_z - 1

        # --- Entrenamiento por K fold ---
        for i in tqdm(range(k), desc="K-Folds Progress"):
            # 1. Separar datos en train y test
            test_idx = kf_idx[i]
            train_idx = np.concatenate([kf_idx[j] for j in range(k) if j != i])
            X_train, z_train = X_scaled[train_idx], z_scaled[train_idx]
            X_test, z_test = X_scaled[test_idx], z_scaled[test_idx]

            # 2. Inicializar el perceptrón con el total de épocas
            perceptron = NonLinearPerceptron(
                learning_rate=nonlinear_learn_rate,
                epochs=epochs,
                beta=beta,
                epsilon=epsilon
            )

            # 3. Entrenar
            errors_train, predictions_per_epoch, errors_test, predictions_per_epoch_test = perceptron.train_and_test(X_train, z_train, X_test, z_test)

            # 4. Desescalar valores
            errors_unscaled = []
            for epoch_preds in predictions_per_epoch:   
                y_pred_scaled = np.array(epoch_preds)
                y_pred_real = (y_pred_scaled + 1) / 2 * range_z + min_z
                z_real = (z_train + 1) / 2 * range_z + min_z
                mse_real = np.mean((y_pred_real - z_real)**2)
                errors_unscaled.append(mse_real)
            errors_test_unscaled = []
            for epoch_preds in predictions_per_epoch_test:   
                y_pred_scaled = np.array(epoch_preds)
                y_pred_real = (y_pred_scaled + 1) / 2 * range_z + min_z
                z_real = (z_test + 1) / 2 * range_z + min_z
                mse_real = np.mean((y_pred_real - z_real)**2)
                errors_test_unscaled.append(mse_real)
            
            # 5. Guardar curvas
            max_len = epochs

            # Rellenar curva de entrenamiento
            if len(errors_unscaled) < max_len:
                last_error = errors_unscaled[-1]
                # Extender con el último error reportado
                errors_unscaled.extend([last_error] * (max_len - len(errors_unscaled)))
                
            # Rellenar curva de prueba
            if len(errors_test_unscaled) < max_len:
                last_error = errors_test_unscaled[-1]
                # Extender con el último error reportado
                errors_test_unscaled.extend([last_error] * (max_len - len(errors_test_unscaled)))

            train_curves.append(errors_unscaled)
            test_curves.append(errors_test_unscaled)

        avg_train = np.nanmean(train_curves, axis=0)
        avg_test = np.nanmean(test_curves, axis=0)
        all_k_results[k] = (avg_train, avg_test)

        # 6. Graficar plot individual
        color = colors(idx) 
        plt.figure(figsize=(8, 5))
        plt.plot(avg_test, label=f'Test (k={k})', color=color)
        plt.plot(avg_train, '--', label=f'Train (k={k})', color=color)
        plt.title(f'Error MSE por Época - K={k}')
        plt.xlabel("Época")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/ej2_generalize_k_{k}.png")
        plt.close()

    # --- Crear figura para el plot general ---
    plt.figure(figsize=(12, 7))
    for idx, (k, (avg_train, avg_test)) in enumerate(all_k_results.items()):
        color = colors(idx)
        plt.plot(avg_train, 
                 label=f'Train (k={k})', 
                 linestyle='--', 
                 color=color)
        plt.plot(avg_test, 
                 label=f'Test (k={k})', 
                 linestyle='-', 
                 color=color)
    plt.title('Comparación de Error (MSE) por Época y K-fold')
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/ej2_generalize.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()