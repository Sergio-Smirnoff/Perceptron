import os
import sys
import traceback
import numpy as np
from ej3_discriminacion_paridad import ParityMultyPerceptron

# ==================== CONFIG ====================
INPUT_PATH = "TP3-ej3-digitos.txt"
OUT_DIR = "outputs_ej3"
DIGITS_OUTFILE = "digits_outputs.txt"

LEARNING_RATE = 0.01
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
                                      layer_one_size=5, layer_two_size=5, optimization_mode="adam") # descgradient, adam, momentum

        # Entrenar
        print("Iniciando entrenamiento...")
        model.train(X, expected_output)
        print("Entrenamiento finalizado.")

        # Predecir: para cada muestra usamos model.forward_pass para obtener la salida cruda
        preds_label = []
        for i in range(len(X)):
            xi = X[i] #xi es la matriz de bits de cada digito
            
            try:
                pred_label = model.predict(xi) #TODO estamos prediciendo con los mismos valores que usamos para entrenar
            except Exception:
                pred_label = None
            preds_label.append(pred_label)

        # === Guardar resultados ===
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, DIGITS_OUTFILE)
        with open(out_path, "w") as f:
            f.write("digit\texpected_parity\n")
            for i in range(len(X)):
                expected = int(expected_output[i])
                pred_l = preds_label[i]
                f.write(f"{expected}\t{pred_l}\n")
            print(f"expected: {expected_output}")
            print("Predicciones (label):", preds_label)

        print(f"Resultados guardados en: {out_path}")


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

