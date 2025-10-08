import datetime
import os
import sys
import traceback
import numpy as np
from ej3_discriminacion_paridad import ParityMultyPerceptron


# ==================== CONFIG ====================
INPUT_PATH = "multicapa/input/TP3-ej3-digitos.txt"
INPUT_TEST_PATH = "multicapa/input/TP3-ej3-digitos-test-light.txt"
OUT_DIR = "outputs_ej3"
DIGITS_OUTFILE = "digits_outputs.txt"
PARITY_OUTFILE = "parity_output.txt"

LEARNING_RATE = 0.1
EPOCHS = 5000
EPSILON = 1e-4
LAYER_ONE_SIZE = 10
LAYER_TWO_SIZE = 10
OPTIMIZATION_MODE = "descgradient"
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

        N_RUNS = 10
        for run in range(1, N_RUNS + 1):

            # ======================= CARGAR DATOS ============================
            X_clean, y_digits_clean = load_digits_flat("multicapa/input/TP3-ej3-digitos.txt")
            X_noise_light, y_digits_noise_light = load_digits_flat("multicapa/input/TP3-ej3-digitos-test-light.txt")
            X_noise_medium, y_digits_noise_medium = load_digits_flat("multicapa/input/TP3-ej3-digitos-test-medium.txt")
            X_noise_heavy, y_digits_noise_heavy = load_digits_flat("multicapa/input/TP3-ej3-digitos-test-heavy.txt")

            # 4 grupos (no mezclamos adentro)
            X_total = [X_clean,       X_noise_light,       X_noise_medium,       X_noise_heavy]
            y_total = [y_digits_clean, y_digits_noise_light, y_digits_noise_medium, y_digits_noise_heavy]

            # ======================= SPLIT POR GRUPOS =======================
            X_train_groups, y_train_groups, X_test_groups, y_test_groups = k_fold(
                X_total, y_total, train_k=3, seed=run  # seed por reproducibilidad
            )
            print(f"Grupos train: {len(X_train_groups)}, grupos test: {len(X_test_groups)}")

            # ======================= ENTRENAR MODELO (PARIDAD) ==============
            model = ParityMultyPerceptron(
                learning_rate=LEARNING_RATE,
                epochs=50000,
                epsilon=EPSILON,
                layer_one_size=10,
                layer_two_size=10,
                optimization_mode=OPTIMIZATION_MODE
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
                   
        with open(os.path.join(OUT_DIR, "digit_predictions.txt"), "w") as f:
            for digit in range(10):
                acc = np.mean(digit_accuracy_rate[digit]) if digit_accuracy_rate[digit] else 0
                f.write(f"{datetime.datetime.now()};{OPTIMIZATION_MODE};EXPECTED={digit};ACCURACY={acc:.4f};\n")
        with open (os.path.join(OUT_DIR, "parity_predictions.txt"), "w") as f:
            for parity_label in (0, 1):
                acc = np.mean(parity_accuracy_rate[parity_label]) if parity_accuracy_rate[parity_label] else 0
                f.write(f"{datetime.datetime.now()};{OPTIMIZATION_MODE};EXPECTED_PARITY={parity_label};ACCURACY={acc:.4f};\n")

            
            

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

