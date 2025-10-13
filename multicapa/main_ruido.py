
import datetime
import os
import sys
import traceback
import numpy as np
from ej3_discriminacion_paridad import ParityMultyPerceptron
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp

import logging as log

log.basicConfig(level=log.INFO)
# ================== CONSTANTS ====================
INPUT_PATH = "multicapa/input/TP3-ej3-digitos.txt"
INPUT_TEST_PATH = "multicapa/input/TP3-ej3-digitos-test-light.txt"
OUT_DIR = "multicapa/outputs_ej3"
DIGITS_OUTFILE = "multicapa/digits_outputs.txt"
PARITY_OUTFILE = "multicapa/parity_output.txt"

LEARNING_RATE = 0.01
EPOCHS = 5000
EPSILON = 1e-4
LAYER_ONE_SIZE = 15
LAYER_TWO_SIZE = 5
OPTIMIZATION_MODE = "descgradient" # "descgradient" or "momentum" or "adam"
NOISE = 0.1 
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

def make_noise(array, noise_level=0.1):
    """
    Agrega ruido a un array de 0/1.
    noise_level: fracción de bits a invertir (0..1)
    """
    num_elements = len(array)
    num_noisy = int(num_elements * noise_level)
    indices = np.random.choice(num_elements, num_noisy, replace=False)
    for idx in indices:
        array[idx] = 1 - array[idx]  # invierte 0 a 1 o 1 a 0
    return array

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

def noise_variation_run():
    """
    Corre el entrenamiento y testeo variando el nivel de ruido.
    """
    results = []
    detours = []
    noises = np.arange(0, 1.1, 0.1)

    for noise in noises:
        print(f"\n=== Noise Level: {noise:.1f} ===")
        
        # Cargar datos limpios
        X_clean, y = load_digits_flat(find_input_file())
        X = make_noise(X_clean.copy(), noise_level=noise)

        pmp = ParityMultyPerceptron(
            layer_one_size=LAYER_ONE_SIZE,
            layer_two_size=LAYER_TWO_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=1,  # Una época a la vez
            epsilon=EPSILON,
            optimization_mode=OPTIMIZATION_MODE
        )

        epoch_errors = []
        epoch_stds = []

        for epoch in range(EPOCHS):
            pmp.train(X_clean, y)
            
            errors = []
            for xi, yi in zip(X, y): 
                num = pmp.predict(xi)
                errors.append(abs(num - yi)) 
            
            # Métricas
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            
            epoch_errors.append(mean_error)
            epoch_stds.append(std_error)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Mean Error = {mean_error:.4f} ± {std_error:.4f}")
        
        results.append(epoch_errors)
        detours.append(epoch_stds)

    # Plotear resultados
    # plot.figure(figsize=(12, 7))
    # epochs_range = range(EPOCHS)
    
    # for i, noise in enumerate(noises):
    #     plot.errorbar(
    #         epochs_range, 
    #         results[i], 
    #         yerr=detours[i], 
    #         label=f"Noise {noise:.1f}", 
    #         capsize=3,
    #         alpha=0.7,
    #         errorevery=5  # Mostrar error bars cada 5 puntos para claridad
    #     )
    
    # plot.xlabel("Epochs")
    # plot.ylabel("Mean Absolute Error")
    # plot.title("Training Error vs Epochs for Different Noise Levels")
    # plot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plot.grid(True, alpha=0.3)
    # plot.tight_layout()
    # plot.savefig(os.path.join(OUT_DIR, f"noise_variation_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    # plot.show()

    # Plotear resultados
    plot.figure(figsize=(12, 7))
    epochs_range = range(EPOCHS)
    
    for i, noise in enumerate(noises):
        plot.plot(
            epochs_range, 
            results[i], 
            label=f"Noise {noise:.1f}", 
            alpha=0.7,
            linewidth=2
        )
    
    plot.xlabel("Epochs")
    plot.ylabel("Mean Absolute Error")
    plot.title("Training Error vs Epochs for Different Noise Levels")
    plot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plot.grid(True, alpha=0.3)
    plot.tight_layout()
    # plot.savefig(os.path.join(OUT_DIR, f"noise_variation_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plot.show()


def plot_mse_curves(errors_dict, title="MSE por época", smooth_window=None):
    """
    Grafica una curva por cada experimento.
    
    Args:
        errors_dict (dict[str, list[float] | np.ndarray]): 
            {nombre_experimento: errores_por_epoca}
        title (str): título del gráfico
        smooth_window (int | None): si se indica (p.ej. 5), aplica media móvil
                                    para suavizar las curvas.
    """
    plt.figure()
    for label, errs in errors_dict.items():
        errs = np.asarray(errs, dtype=float)
        if smooth_window and smooth_window > 1 and len(errs) >= smooth_window:
            # media móvil simple
            kernel = np.ones(smooth_window) / smooth_window
            errs_sm = np.convolve(errs, kernel, mode='valid')
            xs = np.arange(1, len(errs_sm)+1)
            plt.plot(xs, errs_sm, label=f"{label} (MA{smooth_window})")
        else:
            xs = np.arange(1, len(errs)+1)
            plt.plot(xs, errs, label=label)
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_mse_table(errors_dict, every=10):
    """
    Imprime en consola una tabla con MSE por época para cada experimento.
    
    Args:
        errors_dict (dict[str, list[float] | np.ndarray])
        every (int): cada cuántas épocas mostrar (p.ej., 1 para todas, 10 para saltos de 10)
    """
    # encabezado
    labels = list(errors_dict.keys())
    widths = [max(len("Epoch"), 5)] + [max(len(lbl), 10) for lbl in labels]
    header = f"{'Epoch'.ljust(widths[0])} " + " ".join(lbl.ljust(w) for lbl, w in zip(labels, widths[1:]))
    print(header)
    print("-" * len(header))
    
    max_epochs = max(len(v) for v in errors_dict.values())
    for ep in range(1, max_epochs + 1):
        if (ep == 1) or (ep == max_epochs) or (ep % every == 0):
            row = [str(ep).ljust(widths[0])]
            for lbl, w in zip(labels, widths[1:]):
                errs = errors_dict[lbl]
                if ep <= len(errs):
                    row.append(f"{errs[ep-1]:.6f}".ljust(w))
                else:
                    row.append("-".ljust(w))
            print(" ".join(row))



def main():
    
    # Limpieza opcional de archivos de salida en este run
    for fname in [DIGITS_OUTFILE, PARITY_OUTFILE, "predictions.txt"]:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    # ======================= RUN =====================

    # Cargar datos limpios
    X_clean, y = load_digits_flat(find_input_file())
    # X = make_noise(X_clean.copy(), noise_level=noise)

    # noise modification run with adam
    noise_variation_run()
    # model_sgd = ParityMultyPerceptron(
    #     layer_one_size=LAYER_ONE_SIZE,
    #     layer_two_size=LAYER_TWO_SIZE,
    #     learning_rate=LEARNING_RATE,
    #     epochs=EPOCHS,
    #     epsilon=EPSILON,
    #     optimization_mode="descgradient"
    # )
    # mse_sgd,  errs_sgd  = model_sgd.train(X_clean, y)

    # # 1) imprimir tabla (cada 20 épocas + primera/última)
    # print_mse_table({
    #     "SGD": errs_sgd
    # }, every=20)

    # # 2) graficar curvas (con suavizado opcional de media móvil 5)
    # plot_mse_curves({
    #     "SGD": errs_sgd
    # }, title="Comparación de MSE por época", smooth_window=5)



if __name__ == "__main__":
    main()

