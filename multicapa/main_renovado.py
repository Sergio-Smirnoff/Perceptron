import os
import sys
import numpy as np
from ej3_discriminacion_paridad import ParityMultyPerceptron
import matplotlib.pyplot as plot
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp
import logging as log
from multiLayer import MultiLayer
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

os.makedirs('multicapa/outputs_ej3', exist_ok=True)
log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log.FileHandler('multicapa/outputs_ej3/main.log', mode='w'),
        log.StreamHandler()
    ],
    force=True
)

logger = log.getLogger("Main")
# Deshabilitar completamente los logs de MultiLayer
log.getLogger('MultiLayer').disabled = True
# ==================== CONFIG ====================
INPUT_PATH = "multicapa/input/TP3-ej3-digitos.txt"
INPUT_TEST_PATH = "multicapa/input/TP3-ej3-digitos-test-light.txt"
OUT_DIR = "multicapa/outputs_ej3"
DIGITS_OUTFILE = "digits_outputs.txt"
PARITY_OUTFILE = "parity_output.txt"

LEARNING_RATE = 0.1
EPOCHS = 10000
EPSILON = 1
LAYER_ONE_SIZE = 25
LAYER_TWO_SIZE = 15
OPTIMIZATION_MODE = "edg" # "edg" or "momentum" or "adam"
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

def plot_confusion_matrix(y_true, y_pred, noise_level, save_path):
    """
    Crea y guarda una matriz de confusión como PNG
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Cantidad'})
    
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.title(f'Matriz de Confusión - Ruido: {noise_level:.1f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Matriz de confusión guardada en: {save_path}")

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

def noise_variation_run(perceptron, X_clean, y):
    """
    Corre el entrenamiento y testeo variando el nivel de ruido.
    """
    noise = np.arange(0, 1.1, 0.1)
    errors_list = []
    accuracy_list = []

    for n in noise:
        logger.info(f"Testeando con ruido {n:.1f}...")
        X_noisy = make_noise(X_clean.copy(), noise_level=n)

        # Predecir
        result = perceptron.predict(X_noisy)

        # Calcular métricas
        mae = np.mean(np.abs(result - y))
        # Revisar accuracy
        accuracy = np.mean(result == y) * 100
        
        logger.info(f"MAE con ruido {n:.1f}: {mae:.4f}")
        logger.info(f"Accuracy con ruido {n:.1f}: {accuracy:.2f}%")
        
        errors_list.append(mae)
        accuracy_list.append(accuracy)
        
        # Guardar matriz de confusión
        cm_path = f"{OUT_DIR}/confusion_matrix_noise_{n:.1f}.png"
        plot_confusion_matrix(y, result, n, cm_path)

    # Gráfico de error vs ruido
    plt.figure(figsize=(10, 6))
    plt.plot(noise, errors_list, marker='o', label='MAE', linewidth=2)
    plt.xlabel("Nivel de Ruido", fontsize=12)
    plt.ylabel("Error Absoluto Medio", fontsize=12)
    plt.title("Error vs Nivel de Ruido", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/error_vs_noise.png", dpi=300)
    plt.close()
    
    # Gráfico de accuracy vs ruido
    plt.figure(figsize=(10, 6))
    plt.plot(noise, accuracy_list, marker='o', color='green', label='Accuracy', linewidth=2)
    plt.xlabel("Nivel de Ruido", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Accuracy vs Nivel de Ruido", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/accuracy_vs_noise.png", dpi=300)
    plt.close()
    
    logger.info(f"Gráficos guardados en {OUT_DIR}")

def mse_vs_epochs_run(X_clean, y):
    
    perceptron = MultiLayer(
        layers_array=[35, LAYER_ONE_SIZE, LAYER_TWO_SIZE, 10],
        learning_rate=LEARNING_RATE,
        epochs=1,
        epsilon=EPSILON,
        optimization_mode=OPTIMIZATION_MODE,
        seed=42
    )

    error_entropy_list = []
    error_mse_list = []
    epochs = []
    logger.info("Iniciando entrenamiento para MSE vs Épocas...")
    for epoch in range(EPOCHS):
        error_entropy, error_mse = perceptron.train(X_clean, y)
        logger.debug(f"Época {epoch+1}/{EPOCHS} - Error Entropía: {error_entropy:.6f}, Error MSE: {error_mse:.6f}")
        error_entropy_list.append(error_entropy)
        error_mse_list.append(error_mse)
        epochs.append(epoch)

    logger.info("Entrenamiento completado para MSE vs Épocas.")
    # Gráfico de error vs épocas
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, error_entropy_list, marker='.', label='Error Entropía')
    plt.plot(epochs, error_mse_list, marker='.', label='Error MSE')
    plt.xlabel("Épocas", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.title("Error vs Épocas", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/error_vs_epochs.png", dpi=300)
    plt.close()


def plot_mse_curves(errors_dict, title="MSE por época"):
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
        xs = np.arange(1, len(errs)+1)
        plt.plot(xs, errs, label=label)
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot.savefig(os.path.join(OUT_DIR, f"noise_error_vs_epochs.png"))
    plt.show()

def noises_error_vs_epochs(X_clean, y):

    perceptron = MultiLayer(
        layers_array=[35, LAYER_ONE_SIZE, LAYER_TWO_SIZE, 10],
        learning_rate=LEARNING_RATE,
        epochs=1,
        epsilon=EPSILON,
        optimization_mode=OPTIMIZATION_MODE,
        seed=42
    )

    
    logger.info("Iniciando entrenamiento para MSE vs Épocas...")

    noise_errors = {noise: [] for noise in np.arange(0, 1.1, 0.1)}
    noises = [noise for noise in np.arange(0, 1.1, 0.1)]
    X = [make_noise(X_clean.copy(), noise_level=noise) for noise in noises]

    for epoch in range(EPOCHS):
        error_entropy, error_mse = perceptron.train(X_clean, y)
        for i, noise in enumerate(noises):
            y_pred = perceptron.predict(X[i])   #array de predicciones para X con ruido
            noise_errors[noise].append(np.mean((y - y_pred)**2))

    

    # errors_dict = {noise: [] for noise in np.arange(0, 1.1, 0.1)}
    # # for noise in np.arange(0, 1.1, 0.1):
    # for noise in [0.0]:
    #     X = make_noise(X_clean.copy(), noise_level=noise)

    #     # noise modification run with adam
    #     model_sgd = MultiLayer(
    #         layers_array=[35, LAYER_ONE_SIZE, LAYER_TWO_SIZE, 10],
    #         learning_rate=LEARNING_RATE,
    #         epochs=1,
    #         epsilon=EPSILON,
    #         optimization_mode="descgradient",
    #         loss_function="mse"
    #     )
    #     for epoch in range(EPOCHS):
    #         mse_sgd,  errs_sgd  = model_sgd.train(X_clean, y)
    #         y_pred = model_sgd.predict(X)   #array de predicciones para X con ruido
    #         errors_dict[noise].append(np.mean((y - y_pred)**2))

    plot_mse_curves(noise_errors, title="Comparación de MSE por época")

def error_vs_epochs_run(X_clean, y):
    perceptron = MultiLayer(
        layers_array=[35, LAYER_ONE_SIZE, LAYER_TWO_SIZE, 10],
        learning_rate=LEARNING_RATE,
        epochs=1,
        epsilon=EPSILON,
        optimization_mode=OPTIMIZATION_MODE,
        seed=42
    )

    error_entropy_list = []
    error_mse_list = []
    epochs = []
    logger.info("Iniciando entrenamiento para Error vs Épocas...")
    for epoch in range(EPOCHS):
        error_entropy, error_mse = perceptron.train(X_clean, y)
        logger.debug(f"Época {epoch+1}/{EPOCHS} - Error Entropía: {error_entropy:.6f}, Error MSE: {error_mse:.6f}")
        error_entropy_list.append(error_entropy)
        error_mse_list.append(error_mse)
        epochs.append(epoch)

    logger.info("Entrenamiento completado para Error vs Épocas.")
    # Gráfico de error vs épocas
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, error_entropy_list, label='Error Entropía')
    plt.plot(epochs, error_mse_list, label='Error MSE')
    plt.xlabel("Épocas", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.title("Error vs Épocas", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/error_vs_epochs.png", dpi=300)
    plt.close()


def main():
    try:

        logger.info("Inicializando ploting...")

        X_clean, y = load_digits_flat(find_input_file())

        # perceptron = MultiLayer(
        #     layers_array=[35, LAYER_ONE_SIZE, LAYER_TWO_SIZE, 10],
        #     learning_rate=LEARNING_RATE,
        #     epochs=EPOCHS,
        #     epsilon=EPSILON,
        #     optimization_mode=OPTIMIZATION_MODE,
        #     seed=42
        # )

        # logger.info("Entrenando modelo...")
        # error_entropy, error_mse = perceptron.train(X_clean, y)
        # logger.info(f"Error mínimo alcanzado (entropía): {error_entropy:.6f}")
        # logger.info(f"Error mínimo alcanzado (MSE): {error_mse:.6f}")

        # # for i in range(10):
        # #     expected = i
        # #     predicted = perceptron.predict(X[i])
        # #     logger.info(f"Esperado: {expected}, Predicho: {predicted}")

        # logger.info("Testeando modelo...")
        # noise_variation_run(perceptron=perceptron, X_clean=X_clean, y=y)
        # logger.info("Finalizando testeo...")

        #mse_vs_epochs_run(X_clean, y)
        error_vs_epochs_run(X_clean, y)
        #noises_error_vs_epochs(X_clean, y)

    except Exception as e:
        logger.error("Ocurrió un error en main: %s", e)


if __name__ == "__main__":
    main()
