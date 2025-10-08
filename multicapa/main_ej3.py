import datetime
import os
import sys
import traceback
import numpy as np
from ej3_discriminacion_paridad import ParityMultyPerceptron

# ==================== CONFIG ====================
INPUT_PATH = "TP3-ej3-digitos.txt"
INPUT_TEST_PATH = "input/TP3-ej3-digitos-test-light.txt"
OUT_DIR = "outputs_ej3"
DIGITS_OUTFILE = "digits_outputs.txt"
OPTIMIZATION_MODE = "adam" # descgradient, adam, momentum

LEARNING_RATE = 0.0001
EPOCHS = 50000
EPSILON = 1e-4
LAYER_ONE_SIZE = 10
LAYER_TWO_SIZE = 10
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


def main():
    try:
        if os.path.exists("outputs_ej3/" + DIGITS_OUTFILE):
            os.system(f"rm -f outputs_ej3/{DIGITS_OUTFILE}")
        predict_per_lable =[[],[],[],[],[],[],[],[],[],[]]
        for j in range(10):
            input_file = find_input_file()
            print("Usando archivo:", input_file)

            # === Cargar datos ===
            X, y_digits = load_digits_flat(input_file)
            # N, D = X.shape
            for i in range(len(X)):
                print(f'Dígito {i}: {X[i]}')

            # Etiquetas de paridad: even -> 1, odd -> -1 (coincide con predict devuelto por la clase)
            # y_parity = np.where((y_digits % 2) == 0, 1.0, -1.0).reshape(-1, 1)

            expected_output = y_digits # digits

            # Instanciar el perceptrón (usa la clase que pegaste arriba)
            model = ParityMultyPerceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS, epsilon=EPSILON, 
                                        layer_one_size=5, layer_two_size=5, optimization_mode=OPTIMIZATION_MODE) 
            # Entrenar
            print(f"Iniciando entrenamiento {j+1}...")
            model.train(X, expected_output)
            print(f"Entrenamiento finalizado {j+1}.")

            # Predecir: para cada muestra usamos model.forward_pass para obtener la salida cruda
            print(f"Realizando predicciones , iteracion {j+1}...")
            test_file = find_test_file()
            print("Usando archivo de test:", test_file)
            X, y_digits = load_digits_flat(test_file)
            preds_label = []
            for i in range(len(X)):
                xi = X[i]
                try:
                    pred_label = model.predict(xi)
                except Exception:
                    pred_label = None
                predict_per_lable[i].append(pred_label)
                preds_label.append(pred_label)

            # === Guardar resultados ===
            os.makedirs(OUT_DIR, exist_ok=True)
            out_path = os.path.join(OUT_DIR, DIGITS_OUTFILE)
            with open(out_path, "a") as f:
                if j == 0:
                    f.write(f"n;digit;expected_parity;{OPTIMIZATION_MODE}\n")
                for i in range(len(X)):
                    expected = int(expected_output[i])
                    pred_l = preds_label[i]
                    f.write(f"{j};{expected};{pred_l}\n")
                print(f"expected: {expected_output}")
                print("Predicciones (label):", preds_label)

                print(f"Resultados guardados en: {out_path}")
        with open(f"outputs_ej3/predictions.txt", "a") as f:
            for i in range(len(predict_per_lable)):
                counter = 0
                for j in range(len(predict_per_lable[i])):
                    if predict_per_lable[i][j] == i:
                        counter += 1
                f.write(f"{datetime.datetime.now()};{OPTIMIZATION_MODE};{i};{counter/10}\n")

            


        #========Prediccion de Paridad========
        print("\nPredicción de paridad para dígitos 0-9:")
        predictions = []
        for digit in range(10):
            digit_bits = X[digit]
            try:
                parity_pred = model.predict_parity(digit_bits)
                parity_str = "Par" if parity_pred == True else "Impar"
                predictions.append((digit, parity_str))
            except Exception:
                parity_str = "Error en predicción"

        # Mostrar todas las predicciones al final
        for digit, parity in predictions:
            print(f"Dígito {digit}: {parity}")
        with open("parity.txt", "a") as f:
            for digit, parity in predictions:
                f.write(f"Dígito {digit}: {parity}\n")
    except Exception:
        print("Ocurrió un error en main:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

