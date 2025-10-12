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

def test_perceptron(learn_rate, epochs, epsilon, X, y, beta=1.0, k_folds=10, seed=0, verbose=False):
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs")

    folds = k_fold_indices(len(X), k_folds, seed=seed)
    #test_errors = []

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
        with open(train_log_path, 'w') as f_train, open(test_log_path, 'w') as f_test:
            perceptron.train(
                X_train, y_train,
                X_test=X_test, y_test=y_test,
                train_log_file=f_train,
                test_log_file=f_test,
                verbose=False
            )

        # 5. Calcular error final
        #preds = perceptron.predict(X_test)
        #final_mse = np.mean((preds - y_test) ** 2)
        #test_errors.append(final_mse)

        #denom_X = (perceptron.X_max - perceptron.X_min)
        #denom_X[denom_X == 0] = 1e-9


    #avg_mse = np.mean(test_errors)
    #print(f"\nPromedio de MSE final en test (perceptrón): {avg_mse:.6f}")
    #return avg_mse

def pad_curves(curves, target_len):
    padded = []
    for curve in curves:
        if len(curve) < target_len:
            pad_len = target_len - len(curve)
            curve = curve + [np.nan] * pad_len
        padded.append(curve)
    return padded

def main():
    ks = [2, 4, 7, 14]
    learn_rate, epochs, epsilon, beta, input_file, _ = parse_params("params.json")
    X, y = parse_training_data(input_file)

    # Calcular promedio del MSE por época
    all_avg_trains = []
    all_avg_tests = []
    labels = []

    for k in ks:
        print(f"\n== Ejecutando experimento con k={k} ==")

        # Eliminar y recrear logs para cada k
        if os.path.exists("logs"):
            shutil.rmtree("logs")
        os.makedirs("logs")

        test_perceptron(learn_rate, epochs, epsilon, X, y, beta=beta, k_folds=k, seed=random.randint(0, 10000))

        train_curves = []
        test_curves = []

        for i in range(k):
            train_path = f"logs/fold_{i}_log.txt"
            test_path = f"logs/fold_{i}_test_log.txt"

            # Leer mse de entrenamiento
            with open(train_path) as f:
                train_mse = [float(line.strip().split(',')[-1]) for line in f]
                train_curves.append(train_mse)

            # Leer mse de test
            with open(test_path) as f:
                test_mse = [float(line.strip()) for line in f]
                test_curves.append(test_mse)

        # Pad con NaNs para igualar longitud
        max_len = max(max(len(c) for c in train_curves), max(len(c) for c in test_curves))
        train_curves = pad_curves(train_curves, max_len)
        test_curves = pad_curves(test_curves, max_len)

        avg_train = np.nanmean(train_curves, axis=0)
        avg_test = np.nanmean(test_curves, axis=0)

        all_avg_trains.append(avg_train)
        all_avg_tests.append(avg_test)
        labels.append(k)

        # === Graficar curva individual para este k ===
        plt.figure(figsize=(8, 5))
        color = plt.get_cmap('tab10')(ks.index(k) % 10)

        plt.plot(avg_train, label=f"Train [k={k}]", color=color, linestyle='-')
        plt.plot(avg_test, label=f"Test [k={k}]", color=color, linestyle='--')

        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.title(f'Curva de MSE para k={k}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"gen_mse_k{k}.png")
        plt.close()


    # === Graficar todas las curvas ===
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')  
    unique_ks = sorted(set(labels))  
    color_dict = {k: cmap(i % 10) for i, k in enumerate(unique_ks)}
    for avg_train, avg_test, k in zip(all_avg_trains, all_avg_tests, labels):
        color = color_dict[k]
        plt.plot(avg_train, label=f"Train [k={k}]", color=color, linestyle='--')
        plt.plot(avg_test, label=f"Test [k={k}]", color=color, linestyle='-')

    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.title('Promedio de MSE por época')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gen_mse.png")
    plt.show()


if __name__ == "__main__":
    main()