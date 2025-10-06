import os
import sys
import traceback
import numpy as np
from ej3_discriminacion_paridad import ParityMultyPerceptron

INPUT_PATH = "multicapa/TP3-ej3-digitos.txt"
OUT_DIR = "outputs_ej3"
PARITY_OUTFILE = "parity_outputs.txt"

LEARNING_RATE = 0.1
EPOCHS = 1000
EPSILON = 1e-6

def find_input_file():
    # intenta rutas por defecto y luego argv[1]
    if os.path.exists(INPUT_PATH):
        return INPUT_PATH
    else:
        print("'TP3-ej3-digitos.txt' no está")
        sys.exit(1)



def load_digits_flat(path):
    """
    Lee el archivo en bloques de 7 líneas x 5 columnas (0/1).
    Retorna X: ndarray (N_digitos, 35) y y_digits (0..N-1)
    """
    with open(path, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip() != ""]

    if len(raw_lines) % 7 != 0:
        raise ValueError(f"Formato inesperado: {len(raw_lines)} líneas no es múltiplo de 7.")

    num_digits = len(raw_lines) // 7
    digits = []
    for i in range(num_digits):
        block = raw_lines[i*7:(i+1)*7]
        flat = []
        for row in block:
            parts = [p for p in row.split() if p in ("0", "1")]
            flat.extend([int(x) for x in parts])
        if len(flat) != 35:
            raise ValueError(f"Bloque {i} tiene {len(flat)} valores (esperaba 35).")
        digits.append(np.array(flat, dtype=float))
    X = digits
    y_digits = np.arange(num_digits)
    return X, y_digits


def main():
    try:
        input_file = find_input_file()
        print("Usando archivo:", input_file)

        X, y_digits = load_digits_flat(input_file)
        # N, D = X.shape
        print(f"Cargados {len(X)} dígitos. Cada entrada tiene {X} características (35).")

        # Etiquetas de paridad: even -> 1, odd -> -1 (coincide con predict devuelto por la clase)
        y_parity = np.where((y_digits % 2) == 0, 1.0, -1.0).reshape(-1, 1)

        # Instanciar el perceptrón (usa la clase que pegaste arriba)
        model = ParityMultyPerceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS, epsilon=EPSILON, 
                                      layer_one_size=1, layer_two_size=1, optimization_mode="descgradient")

        # Entrenar
        print("Iniciando entrenamiento...")
        model.train(X, y_parity)
        print("Entrenamiento finalizado.")

        # Predecir: para cada muestra usamos model.forward_pass para obtener la salida cruda
        preds_label = []
        preds_raw = []
        for i in range(N):
            xi = X[i]
            try:
                acts = model.forward_pass(xi)
                raw_out = acts[-1]
            except Exception:
                # si forward_pass falla por la forma de entrada, intentamos pasar como vector fila
                try:
                    acts = model.forward_pass(xi.reshape(1, -1))
                    raw_out = acts[-1]
                except Exception:
                    raw_out = None
            # Si el método predict está disponible (devuelve 1/-1), lo usamos
            try:
                pred_label = model.predict(xi)
            except Exception:
                pred_label = None

            preds_raw.append(raw_out)
            preds_label.append(pred_label)

        # Crear directorio y escribir archivo de salida
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, PARITY_OUTFILE)
        with open(out_path, "w") as f:
            f.write("digit\texpected_parity\tpred_label\tpred_raw\n")
            for i in range(N):
                expected = int(y_parity[i, 0])
                pred_l = preds_label[i]
                raw = preds_raw[i]
                # normalizar representación de 'raw' para escribir
                raw_str = ""
                try:
                    if raw is None:
                        raw_str = "None"
                    elif isinstance(raw, np.ndarray):
                        # convertir a lista y truncar si largo
                        raw_list = raw.tolist()
                        raw_str = ",".join(f"{float(x):.6f}" for x in np.ravel(raw_list))
                    else:
                        raw_str = str(raw)
                except Exception:
                    raw_str = repr(raw)
                f.write(f"{i}\t{expected}\t{pred_l}\t{raw_str}\n")

        print(f"Resultados guardados en: {out_path}")

    except Exception as e:
        print("Ocurrió un error en main:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

