import datetime
import os
import sys
import traceback
import numpy as np
from ej3_discriminacion_paridad import ParityMultyPerceptron
import matplotlib.pyplot as plot
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp


# ==================== CONFIG ====================
INPUT_PATH = "multicapa/input/TP3-ej3-digitos.txt"
INPUT_TEST_PATH = "multicapa/input/TP3-ej3-digitos-test-light.txt"
OUT_DIR = "outputs_ej3"
DIGITS_OUTFILE = "digits_outputs.txt"
PARITY_OUTFILE = "parity_output.txt"

LEARNING_RATE = 0.0001
EPOCHS = 50000
EPSILON = 1e-6
LAYER_ONE_SIZE = 5
LAYER_TWO_SIZE = 5
OPTIMIZATION_MODE = "adam"
# =================================================

def find_input_file():
    """Busca el archivo de entrada."""
    if os.path.exists(INPUT_PATH):
        return INPUT_PATH
    else:
        print(f"'{INPUT_PATH}' no está en el directorio actual.")
        sys.exit(1)

def find_test_file():
    """Busca el archivo de entrada de test."""
    if os.path.exists(INPUT_TEST_PATH):
        return INPUT_TEST_PATH
    else:
        print(f"'{INPUT_TEST_PATH}' no está en el directorio actual.")
        sys.exit(1)


def load_digits_flat(path):
    """
    Lee el archivo en bloques de 7x5 (0/1).
    Retorna X (N_digitos, 35) y etiquetas y_digits (0..N-1).
    """
    with open(path, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip() != ""]

    if len(raw_lines) % 7 != 0:
        raise ValueError(f"Formato inesperado: {len(raw_lines)} líneas no es múltiplo de 7.")

    num_digits = len(raw_lines) // 7
    digits = []
    for i in range(num_digits):
        block = raw_lines[i * 7:(i + 1) * 7]
        flat = []
        for row in block:
            parts = [p for p in row.split() if p in ("0", "1")]
            flat.extend([int(x) for x in parts])
        if len(flat) != 35:
            raise ValueError(f"Bloque {i} tiene {len(flat)} valores (esperaba 35).")
        digits.append(np.array(flat, dtype=int))
    X = digits
    y_digits = np.arange(num_digits)
    return X, y_digits


def k_fold(X_list, y_list, train_k=3, seed=None):
    """
    Selecciona 'train_k' grupos completos para entrenamiento
    y deja el resto como test.
    
    X_list: lista de grupos (cada grupo = lista/matriz de bits)
    y_list: lista de etiquetas (alineadas 1 a 1 con X_list)
    train_k: cantidad de grupos para entrenamiento
    seed: int opcional para reproducibilidad
    """
    assert len(X_list) == len(y_list), "X_list y y_list deben tener misma longitud"
    n_groups = len(X_list)
    assert 1 <= train_k < n_groups, "train_k debe ser >=1 y < cantidad de grupos"

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_groups)
    train_idx = rng.choice(all_idx, size=train_k, replace=False)
    test_idx = np.setdiff1d(all_idx, train_idx)

    X_train = [X_list[i] for i in train_idx]
    y_train = [y_list[i] for i in train_idx]
    X_test  = [X_list[i] for i in test_idx]
    y_test  = [y_list[i] for i in test_idx]
    return X_train, y_train, X_test, y_test

def save_digits_results(run_id, y_true_digits, y_pred_digits, out_dir, filename="digits_results.csv", tasa_acierto=0):
    """
    Formato requerido: n;digit;expected_digit;error
    - digit: predicho
    - expected_digit: ground truth
    - error: expected_digit - digit
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    header_needed = not os.path.exists(path)

    with open(path, "a") as f:
        if header_needed:
            f.write("n;     digit;      expected_digit;     error:      tasa_acierto:\n")
        for yd, yp in zip(y_true_digits, y_pred_digits):
            f.write(f"{run_id}; {int(yp)}; {int(yd)};  {int(yd) - int(yp)}; {tasa_acierto}\n")


def save_parity_results(run_id, y_true_digits, y_pred_parity, out_dir, filename="parity_results.csv", tasa_acierto=0):
    """
    Formato: n;digit;expected_parity;error
    - digit: el dígito real (0..9) para referencia
    - error: expected_parity - predicted_parity
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    header_needed = not os.path.exists(path)

    with open(path, "a") as f:
        if header_needed:
            f.write("n; digit;  expected_parity;    error:   tasa_acierto:\n")
        for yd, yp in zip(y_true_digits, y_pred_parity):
            f.write(f"{run_id}; {int(yd)};      {yp};     {yd - int(yp)}; {tasa_acierto}\n")

def default_run(
        X_total, 
        y_total, 
        digit_accuracy_rate, 
        parity_accuracy_rate, 
        learning_rate=LEARNING_RATE, 
        epochs=EPOCHS,
        optimization_mode=OPTIMIZATION_MODE,
        epsilon=EPSILON
        ):
    N_RUNS = 5
    for run in range(1, N_RUNS + 1):
        # ======================= SPLIT POR GRUPOS =======================
        X_train_groups, y_train_groups, X_test_groups, y_test_groups = k_fold(
            X_total, y_total, train_k=3, seed=run  # seed por reproducibilidad
        )
        print(f"Grupos train: {len(X_train_groups)}, grupos test: {len(X_test_groups)}")
        # ======================= ENTRENAR MODELO (PARIDAD) ==============
        model = ParityMultyPerceptron(
            learning_rate=learning_rate,
            epochs=epochs,
            epsilon=epsilon,
            layer_one_size=10,
            layer_two_size=10,
            optimization_mode=optimization_mode
        )
        print(f"Entrenando, run {run}...")
        for X, y in zip(X_train_groups, y_train_groups):  
            model.train(X, y)  
        print(f"Entrenamiento finalizado run {run}.")
        # Métricas simples
        for X_test, y_test in zip(X_test_groups, y_test_groups):
            for x, y in zip(X_test, y_test):
                digit_accuracy_rate[y].append(1 if model.predict(x) == y else 0)
                parity_accuracy_rate[y%2].append(1 if model.predict_parity(x) else 0)
    return digit_accuracy_rate, parity_accuracy_rate

def lr_run(
        learning_rate=LEARNING_RATE, 
        epochs=2000,
        optimization_mode=OPTIMIZATION_MODE,
        epsilon=EPSILON
    ):

    model = ParityMultyPerceptron(
        learning_rate=learning_rate,
        epochs=5,
        epsilon=epsilon,
        layer_one_size=10,
        layer_two_size=10,
        optimization_mode=optimization_mode
    )

    X, y = load_digits_flat("multicapa/input/TP3-ej3-digitos.txt")
    print(f"Entrenando con learning rate {learning_rate}...")

    epoch = []
    accuracies = []
    
    # Accuracy inicial (antes de entrenar) - promedio de 20 evaluaciones
    correct_sum = 0
    num_evaluations = 20
    
    for j in range(num_evaluations):
        # Predecir para CADA muestra individual
        correct_count = 0
        for xi, yi in zip(X, y):
            result = model.predict_parity(xi)
            expected_parity = (yi % 2 == 0)
            if result == expected_parity:
                correct_count += 1
        
        # Calcular el accuracy de esta evaluación
        eval_accuracy = correct_count / len(X)
        correct_sum += eval_accuracy
    
    # Promedio de las 20 evaluaciones
    accuracy = correct_sum / num_evaluations
    epoch.append(0)
    accuracies.append(accuracy)

    # Entrenar por cada época
    for ep in range(1, epochs):
        model.train(X, y)
        
        correct_sum = 0
        for j in range(num_evaluations):
            correct_count = 0
            for xi, yi in zip(X, y):
                result = model.predict_parity(xi)
                expected_parity = (yi % 2 == 0)
                if result == expected_parity:
                    correct_count += 1
            
            eval_accuracy = correct_count / len(X)
            correct_sum += eval_accuracy
        
        accuracy = correct_sum / num_evaluations
        epoch.append(ep*5)
        accuracies.append(accuracy)

    print(f"Learning rate {learning_rate} completado. Accuracy final: {accuracy:.4f}")
    return learning_rate, epoch, accuracies


def run_single_lr(lr):
    """Función auxiliar para ejecutar un solo learning rate"""
    return lr_run(learning_rate=lr)


def lr_variation_run():
    
    acc_mapper = {
        0.1: [],
        0.01: [],
        0.001: [],
        0.0001: []
    }

    epoch_mapper = {
        0.1: [],
        0.01: [],
        0.001: [],
        0.0001: []
    }
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    # Usar multiprocessing para ejecutar en paralelo
    with Pool(processes=min(len(learning_rates), mp.cpu_count())) as pool:
        results = pool.map(run_single_lr, learning_rates)
    
    # Procesar los resultados
    for lr, epochs, accuracies in results:
        epoch_mapper[lr] = epochs
        acc_mapper[lr] = accuracies

    # Plotear
    plot.figure(figsize=(10, 6))
    for lr in learning_rates:
        plot.plot(epoch_mapper[lr], acc_mapper[lr], marker='o', markersize=3, linewidth=2, label=f'LR={lr}')
    plot.xlabel('Épocas')
    plot.ylabel('Accuracy (promedio de 20 evaluaciones)')
    plot.title('Accuracy de Paridad vs Épocas para distintos Learning Rates')
    plot.grid(True)
    plot.legend()
    plot.savefig('multicapa/outputs_ej3/parity_accuracy_vs_epochs_lr_variation.png')
    plot.show()

def run_single_epoch(args):
    """Función helper para ejecutar un experimento con un número de épocas"""
    ep, X_total, y_total = args
    
    digit_accuracy_rate = [[] for _ in range(10)]
    parity_accuracy_rate = [[], []]  
    digit_accuracy_rate, parity_accuracy_rate = default_run(
        X_total, y_total, digit_accuracy_rate, parity_accuracy_rate, epochs=ep
    )
    
    # Calcular media y desviación estándar
    acc_per_digit = [np.mean(digit_accuracy_rate[i]) for i in range(10)]
    acc_total = np.mean(acc_per_digit)
    acc_std = np.std(acc_per_digit)
    
    return ep, acc_total, acc_std


# variacion de epochs manteniendo learning rate, epsilon y optimizador fijo
def epochs_variation_run(X_total, y_total):
    epoch_list = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 100000]
    args_list = [(ep, X_total, y_total) for ep in epoch_list]
    num_cores = mp.cpu_count()
    print(f"Usando {num_cores} núcleos")
    
    with Pool(processes=num_cores) as pool:
        results = pool.map(run_single_epoch, args_list)
    
    results.sort(key=lambda x: x[0])
    
    epoch_list_sorted = [r[0] for r in results]
    accuracy_list = [r[1] for r in results]
    std_list = [r[2] for r in results]
    
    # Plotear
    plot.figure(figsize=(10, 6))
    plot.errorbar(epoch_list_sorted, accuracy_list, yerr=std_list, marker='o', 
                  linewidth=2, capsize=5, capthick=2, label='Accuracy promedio')
    plot.xlabel('Épocas')
    plot.ylabel('Accuracy Total Promedio')
    plot.title('Accuracy vs Número de Épocas')
    plot.grid(True)
    plot.legend()
    plot.savefig('multicapa/outputs_ej3/accuracy_vs_epochs.png')
    plot.show()


# variacion de epsilon manteniendo learning rate, epochs y optimizador fijo
def epsilon_variation_run(X_total, y_total, digit_accuracy_rate, parity_accuracy_rate):
    for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        default_run(X_total, y_total, digit_accuracy_rate, parity_accuracy_rate, epsilon=eps)

def main():
    try:
        # Limpieza opcional de archivos de salida en este run
        for fname in [DIGITS_OUTFILE, PARITY_OUTFILE, "predictions.txt"]:
            fpath = os.path.join(OUT_DIR, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

        # Para métricas simples por etiqueta de paridad
        predict_per_label = { -1: 0, 1: 0 }

        digit_accuracy_rate = [[] for _ in range(10)]  # para cada dígito, lista de aciertos (1) o errores (0)
        parity_accuracy_rate = [[], []]  # para cada etiqueta de paridad, lista de aciertos (1) o errores (0)

        # ======================= CARGAR DATOS ============================
        X_clean, y_digits_clean = load_digits_flat("multicapa/input/TP3-ej3-digitos.txt")
        X_noise_light, y_digits_noise_light = load_digits_flat("multicapa/input/TP3-ej3-digitos-test-light.txt")
        X_noise_medium, y_digits_noise_medium = load_digits_flat("multicapa/input/TP3-ej3-digitos-test-medium.txt")
        X_noise_heavy, y_digits_noise_heavy = load_digits_flat("multicapa/input/TP3-ej3-digitos-test-heavy.txt")

        # 4 grupos (no mezclamos adentro)
        X_total = [X_clean,       X_noise_light,       X_noise_medium,       X_noise_heavy]
        y_total = [y_digits_clean, y_digits_noise_light, y_digits_noise_medium, y_digits_noise_heavy]

        # ======================= RUN =====================
        # default_run(X_total, y_total, digit_accuracy_rate, parity_accuracy_rate)
        #epochs_variation_run(X_total, y_total)
        # epsilon_variation_run(X_total, y_total, digit_accuracy_rate, parity_accuracy_rate)
        lr_variation_run()   
            

        # Resumen por etiqueta de paridad
        with open(os.path.join(OUT_DIR, "predictions.txt"), "a") as f:
            total_preds = sum(predict_per_label.values()) or 1
            for label in (-1, 1):
                frac = predict_per_label[label] / total_preds
                f.write(f"{datetime.datetime.now()};{OPTIMIZATION_MODE};parity={label};{frac:.4f}\n")

    except Exception as e:
        print("Ocurrió un error en main:", e)



if __name__ == "__main__":
    main()
