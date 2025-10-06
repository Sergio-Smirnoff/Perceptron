import os
import sys
import traceback
import numpy as np
from ej3_discriminacion_paridad import ParityMultyPerceptron

# ==================== CONFIG ====================
INPUT_PATH = "TP3-ej3-digitos.txt"
OUT_DIR = "outputs_ej3"
DIGITS_OUTFILE = "digits_outputs.txt"

LEARNING_RATE = 0.1
EPOCHS = 50000
EPSILON = 1e-6
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
        digits.append(np.array(flat, dtype=float))
    X = np.vstack(digits)
    y_digits = np.arange(num_digits)
    return X, y_digits


def main():
    try:
        input_file = find_input_file()
        print("Usando archivo:", input_file)

        # === Cargar datos ===
        X, y_digits = load_digits_flat(input_file)
        N, D = X.shape
        print(f"Cargados {N} dígitos. Cada entrada tiene {D} características (35 bits por número).")

        # === Crear modelo ===
        model = ParityMultyPerceptron(
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            epsilon=EPSILON,
            layer_one_size=LAYER_ONE_SIZE,
            layer_two_size=LAYER_TWO_SIZE
        )

        # === Entrenar modelo ===
        print("Entrenando red neuronal para identificación de números...")
        model.train(X, y_digits)
        print("Entrenamiento finalizado.\n")

        # === Predicciones ===
        preds = []
        for i in range(N):
            pred = model.predict(X[i])
            preds.append(pred)

        # === Guardar resultados ===
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, DIGITS_OUTFILE)
        with open(out_path, "w") as f:
            f.write("expected_digit\tpredicted_digit\n")
            for i in range(N):
                expected = int(y_digits[i])
                pred = preds[i]
                f.write(f"{expected}\t{pred}\n")

        print(f"Resultados guardados en: {out_path}")

    except Exception:
        print("Ocurrió un error en main:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

