import random

import numpy as np
from LinearPerceptron import LinearPerceptron
import json
import pandas as pd
from tqdm import tqdm
from NonLinearPerceptron import NonLinearPerceptron
from ClassifierPerceptron import ClassifierPerceptron
import matplotlib.pyplot as plt
import shutil
import os

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def parse_params(params):
    with open(params) as f:
        params = json.load(f)
    learn_rate = params["learn_rate"]
    epochs = params["epochs"]
    epsilon = params["epsilon"]
    beta = params["non_linear_beta"]
    input_file = params["input_file"]
    output_file = params["output_file"]
    return learn_rate, epochs, epsilon, beta, input_file, output_file

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


# En ej2_generalization.py

# En ej2_generalization.py

def test_perceptron(learn_rate, epochs, epsilon, X, y, beta=1.0, k_folds=10, seed=0, verbose=False):
    # Crear o limpiar el directorio de logs
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs")

    folds = k_fold_indices(len(X), k_folds, seed=seed)
    test_errors = []
    baseline_errors = []

    # tqdm para ver el progreso de los folds
    for i in tqdm(range(k_folds), desc="K-Folds Progress"):
        # 1. Separar datos en train y test
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 2. Inicializar el perceptrón con el total de épocas
        perceptron = NonLinearPerceptron(
            learning_rate=learn_rate, epochs=epochs, epsilon=epsilon, beta=beta
        )

        # 3. Definir rutas para los logs del fold actual
        train_log_path = f"logs/fold_{i}_log.txt"
        test_log_path = f"logs/fold_{i}_test_log.txt"

        # 4. Entrenar y loguear en una sola llamada
        # Usamos 'with' para manejar los archivos de forma segura
        with open(train_log_path, 'w') as f_train, open(test_log_path, 'w') as f_test:
            perceptron.train(
                X_train, y_train,
                X_test=X_test, y_test=y_test,
                train_log_file=f_train,
                test_log_file=f_test,
                verbose=False  # El verbose del perceptrón no es necesario aquí
            )

        # 5. Calcular error final y baseline (sin cambios en esta parte)
        preds = perceptron.predict(X_test)
        final_mse = np.mean((preds - y_test) ** 2)
        test_errors.append(final_mse)

        # ... (El resto del código de baseline se mantiene igual)
        denom_X = (perceptron.X_max - perceptron.X_min)
        denom_X[denom_X == 0] = 1e-9
        X_train_scaled = (X_train - perceptron.X_min) / denom_X
        X_test_scaled = (X_test - perceptron.X_min) / denom_X
        denom_y = (perceptron.y_max - perceptron.y_min) if (perceptron.y_max - perceptron.y_min) != 0 else 1e-9
        y_train_scaled = (y_train - perceptron.y_min) / denom_y


    avg_mse = np.mean(test_errors)
    print(f"\nPromedio de MSE final en test (perceptrón): {avg_mse:.6f}")
    return avg_mse

def pad_curves(curves, target_len):
    padded = []
    for curve in curves:
        if len(curve) < target_len:
            pad_len = target_len - len(curve)
            curve = curve + [np.nan] * pad_len
        padded.append(curve)
    return padded

def main():
    k = 14 #[2, 4, 7, 14]
    learn_rate, epochs, epsilon, beta, input_file, _ = parse_params("params.json")
    X, y = parse_training_data(input_file)

    test_perceptron(learn_rate, epochs, epsilon, X, y, beta=beta, k_folds=k, seed=random.randint(0, 10000))
    # Calcular promedio del MSE por época
    train_curves = []
    test_curves = []

    for i in range(k):
        train_path = f"logs/fold_{i}_log.txt"
        test_path = f"logs/fold_{i}_test_log.txt"

        # Leer mse de entrenamiento (último valor de cada línea)
        with open(train_path) as f:
            train_mse = [float(line.strip().split(',')[-1]) for line in f]
            train_curves.append(train_mse)

        # Leer mse de test
        with open(test_path) as f:
            test_mse = [float(line.strip()) for line in f]
            test_curves.append(test_mse)

    # === Promedios ignorando NaNs ===
    max_len = max(max(len(c) for c in train_curves), max(len(c) for c in test_curves))
    train_curves = pad_curves(train_curves, max_len)
    test_curves = pad_curves(test_curves, max_len)

    avg_train = np.nanmean(train_curves, axis=0)
    avg_test = np.nanmean(test_curves, axis=0)

    # Graficar
    plt.plot(avg_train, label=f'train [k={k}]')
    plt.plot(avg_test, label=f'test [k={k}]')

    plt.xlabel('Época')
    plt.ylabel('MSE entrenamiento')
    plt.title('Promedio de MSE por época')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gen_mse.png")
    plt.show()


if __name__ == "__main__":
    main()