import datetime
import os
import sys
import traceback
import numpy as np
from ej3_discriminacion_paridad import ParityMultyPerceptron
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


"""
Matriz de confusion multiclase:
por cada cuadro tengo la cantidad de true positives que se tuvieron
si hago 10 corridas: predigo cada valor en loop


"""

# ==================== CONFIG ====================
INPUT_PATH = "multicapa/input/TP3-ej3-digitos.txt"
INPUT_TEST_PATH = "multicapa/input/TP3-ej3-digitos-test-light.txt"
OUT_DIR = "outputs_ej3"
DIGITS_OUTFILE = "digits_outputs.txt"
PARITY_OUTFILE = "parity_output.txt"

LEARNING_RATE = 0.01
EPOCHS = 10000
EPSILON = 1e-4
LAYER_ONE_SIZE = 25
LAYER_TWO_SIZE = 25
OPTIMIZATION_MODE = "descgradient"
# =================================================

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

def confusion_matrix_digits(X_complete, y_complete, perceptron):
    # 5 corridas de testeo sobre TODO el set (o poné X_test si querés solo test)
    y_true_all = []
    y_pred_all = []

    for j in range(5):
        for i, x in enumerate(X_complete):
            p = int(perceptron.predict(np.array(x)))  # debe devolver 0..9
            y_true_all.append(i)                      # etiqueta real (0..9 en ese orden)
            y_pred_all.append(p)                      # etiqueta predicha

    # Aciertos por dígito (índice = dígito)
    hits = [0]*10
    for t, p in zip(y_true_all, y_pred_all):
        if t == p:
            hits[t] += 1

    print(f"Aciertos por dígito (sobre {len(y_true_all)//10} corridas): {hits}")

    # Matriz de confusión (conteos)
    labels = list(range(10))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    print("Matriz de confusión (conteos):\n", cm)

    # Normalizada por clase real (recall por dígito)
    cm_norm = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(values_format='.2f', cmap='Blues', colorbar=True)
    plt.title("Matriz de confusión - normalizada por clase real")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_digits_normalized.png"))

def confusion_matrix_parity(X_complete, y_complete, perceptron):
    # 5 corridas de testeo sobre TODO el set (o poné X_test si querés solo test)
    y_true_all = []
    y_pred_all = []

    for j in range(5):
        for i, x in enumerate(X_complete):
            p = int(perceptron.predict_parity(np.array(x)))  # debe devolver 0 o 1
            parity = True if (i % 2) == 1 else False          # etiqueta real (0=par, 1=impar)
            y_true_all.append(parity)
            y_pred_all.append(p)                      # etiqueta predicha

    # Aciertos por clase (0=par, 1=impar)
    hits = [0, 0]
    for t, p in zip(y_true_all, y_pred_all):
        if t == p:
            hits[t] += 1

    print(f"Aciertos por clase (sobre {len(y_true_all)//2} corridas): {hits}")

    # Matriz de confusión (conteos)
    labels = [0, 1]
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    print("Matriz de confusión (conteos):\n", cm)

    # Normalizada por clase real (recall por clase)
    cm_norm = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(values_format='.2f', cmap='Blues', colorbar=True)
    plt.title("Matriz de confusión - normalizada por clase real")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'confusion_matrix_parity_{EPOCHS}.png'))

def main():
    X_complete, y_complete = load_digits_flat(INPUT_PATH)


    #test group con nro 1 separado
    X_train, y_train, X_test, y_test = k_fold(X_complete, y_complete, train_k=9, seed=420)
    
    perceptron = ParityMultyPerceptron(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        epsilon=EPSILON,
        layer_one_size=LAYER_ONE_SIZE,
        layer_two_size=LAYER_TWO_SIZE,
        optimization_mode=OPTIMIZATION_MODE
    )
    perceptron.train(X_train, y_train)

    # confusion_matrix_digits(X_complete, y_complete, perceptron)

    confusion_matrix_parity(X_complete, y_complete, perceptron)

if __name__ == "__main__":
    main()
