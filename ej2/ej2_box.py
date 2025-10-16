import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from NonLinearPerceptron import NonLinearPerceptron # Asumimos que la clase aún existe

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
    return X, y, df

def k_fold_indices(n_samples, k, seed=0):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)
    return folds

def main():
    # --- Parámetros del Experimento ---
    nonlinear_learn_rate, epochs, epsilon, beta, input_file, k_values = parse_params("config/ej2_box_params.json")
    X_values, z_values, original_df = parse_training_data(input_file)

    # Variables para el plot general
    all_k_fold_mse = {}

    peak_analysis_data = []
    
    # Pre-calculamos los colores para los gráficos de barras (uno por k)
    colors = plt.cm.get_cmap('viridis', len(k_values)) 

    # --- Entrenamiento por K valor ---
    for idx, k in enumerate(k_values):
        k_fold_train_mse = []
        k_fold_test_mse = []

        kf_idx = k_fold_indices(len(X_values), k, seed=42)

        max_mse_for_k = -1.0
        peak_mse_fold_number = -1
        peak_mse_test_indices = None

        # --- Preprocesamiento General (Aplicado a todos los datos para mantener el mismo escalado en todos los folds) ---
        # Escalar X a [-1, 1]
        min_x = X_values.min(axis=0)
        range_x = X_values.max(axis=0) - min_x
        range_x[range_x == 0] = 1 
        X_scaled = 2 * (X_values - min_x) / range_x - 1

        # Escalar z a [-1, 1]
        min_z = z_values.min()
        range_z = z_values.max() - min_z
        range_z = 1 if range_z == 0 else range_z
        z_scaled = 2 * (z_values - min_z) / range_z - 1

        mse_train = []
        mse_test = []
        # --- Entrenamiento por K fold ---
        for i in tqdm(range(k), desc=f"K={k} Folds Progress"):
            # 1. Separar datos en train y test
            test_idx = kf_idx[i]
            train_idx = np.concatenate([kf_idx[j] for j in range(k) if j != i])
            X_train, z_train = X_scaled[train_idx], z_scaled[train_idx]
            X_test, z_test = X_scaled[test_idx], z_scaled[test_idx]

            z_train_raw = z_values[train_idx]
            z_test_raw = z_values[test_idx]

            # 2. Inicializar el perceptrón
            perceptron = NonLinearPerceptron(
                learning_rate=nonlinear_learn_rate,
                epochs=epochs,
                beta=beta,
                epsilon=epsilon
            )

            # 3. Entrenar
            errors_train, predictions_per_epoch, errors_test, predictions_per_epoch_test = perceptron.train_and_test(X_train, z_train, X_test, z_test)

            # 4. Desescalar el MSE final
            
            # Train
            final_train_preds_scaled = predictions_per_epoch[-1]
            y_pred_real_train = (np.array(final_train_preds_scaled) + 1) / 2 * range_z + min_z
            mse_real_train = np.mean((y_pred_real_train - z_train_raw)**2)
            k_fold_train_mse.append(mse_real_train)
            
            # Test
            final_test_preds_scaled = predictions_per_epoch_test[-1]
            y_pred_real_test = (np.array(final_test_preds_scaled) + 1) / 2 * range_z + min_z
            mse_real_test = np.mean((y_pred_real_test - z_test_raw)**2)
            k_fold_test_mse.append(mse_real_test)

            if mse_real_test > max_mse_for_k:
                max_mse_for_k = mse_real_test
                peak_mse_fold_number = i + 1 
                peak_mse_test_indices = test_idx

        peak_analysis_data.append({
            "k": k,
            "peak_fold": peak_mse_fold_number,
            "peak_mse": max_mse_for_k,
            "test_indices": peak_mse_test_indices
        })

        # 5. Guardar resultados para el plot general
        all_k_fold_mse[k] = {"train":k_fold_train_mse, "test":k_fold_test_mse}

        # --- Generar Plot de Barras Individual por K ---
        bar_width = 0.35
        fold_labels = [f'{i+1}' for i in range(k)]
        x = np.arange(k)
        
        # Color asignado al valor de K
        bar_color = colors(idx) 

        plt.figure(figsize=(10, 6))
        
        # Barras de Test
        plt.bar(x + bar_width/2, k_fold_test_mse, bar_width, label='Test MSE Final', color=bar_color)
        
        # Barras de Train
        plt.bar(x - bar_width/2, k_fold_train_mse, bar_width, label='Train MSE Final', color=bar_color, alpha=0.6, hatch='/')
        
        plt.title(f'Variabilidad del MSE Final por Fold (K={k})')
        plt.xlabel("Fold")
        plt.ylabel("MSE Final")
        plt.xticks(x, fold_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(f"plots/ej2_final_mse_k_{k}.png")
        plt.close()

    with open("results/ej2_box_CV.txt", "w") as f:
        header1 = "--- Métrica de Estabilidad del Modelo (Proxy de Homogeneidad de Datos) ---\n"
        header2 = "k \t | Error Prom. (μ) \t | Desv. Est. (σ) \t | Coef. Variación (CV)\n"
        header3 = "----------------------------------------------------------------------------------\n"
        f.write(header1)
        f.write(header2)
        f.write(header3)
        for k, results in all_k_fold_mse.items():
            test_errors = results['test']
            
            mean_mse = np.mean(test_errors)
            std_dev_mse = np.std(test_errors)
            
            if mean_mse == 0:
                cv_mse = 0
            else:
                cv_mse = (std_dev_mse / mean_mse) * 100
            result_line = f"{k} \t | {mean_mse:.4f} \t\t | {std_dev_mse:.4f} \t\t | {cv_mse:.2f}%\n"
            f.write(result_line)

    with open("results/ej2_box_peak.txt", "w") as f:
        f.write("--- Análisis Detallado de los Folds con Mayor Error (MSE) ---\n\n")
        
        for analysis in peak_analysis_data:
            k = analysis["k"]
            fold_num = analysis["peak_fold"]
            mse_val = analysis["peak_mse"]
            indices = analysis["test_indices"]
            
            f.write(f"==================== K = {k} ====================\n")
            f.write(f"Pico de MSE encontrado en el Fold: {fold_num}\n")
            f.write(f"Valor del Pico de MSE: {mse_val:.4f}\n\n")
            f.write(f"Datos en el conjunto de prueba de este fold (Índices originales):\n")
            
            # Usamos el DataFrame original para mostrar los datos
            problematic_data = original_df.iloc[indices]
            f.write(problematic_data.to_string())
            f.write(f"\n\n")

    # --- Generar Plot de Barras General (Test) ---
    plt.figure(figsize=(15, 7))
    total_folds = 0
    bar_positions = []
    bar_values = []
    bar_colors = []
    bar_labels = []

    # 1. Recolectar todos los datos de Test para el plot general
    for idx, k in enumerate(k_values):
        test_mse = all_k_fold_mse[k]['test']
        num_folds = len(test_mse)
        
        # Posiciones de las barras: agregamos un espacio entre los grupos K
        start_pos = total_folds + (idx * 0.5) 
        current_positions = np.arange(start_pos, start_pos + num_folds)
        
        bar_positions.extend(current_positions)
        bar_values.extend(test_mse)
        bar_colors.extend([colors(idx)] * num_folds)
        bar_labels.extend([f'K{k} F{i+1}' for i in range(num_folds)])
        
        total_folds += num_folds

    # 2. Plotear
    plt.bar(bar_positions, bar_values, color=bar_colors, width=0.4)

    plt.title('Comparación de la Variabilidad del MSE de Prueba Final por K')
    plt.xlabel("Fold (Agrupado por K)")
    plt.ylabel("MSE Final de Prueba")
    plt.xticks(bar_positions, bar_labels, rotation=45, ha='right')

    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig("plots/ej2_final_mse_general.png")
    plt.close()

if __name__ == '__main__':
    main()