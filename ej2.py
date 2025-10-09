import numpy as np
from LinearPerceptron import LinearPerceptron
import json
import pandas as pd
from tqdm import tqdm
from NonLinearPerceptron import NonLinearPerceptron
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def read_mse_from_log(log_file):
    """
    Lee los valores de MSE desde un archivo de log.
    Retorna una lista con los MSE por época.
    """
    mse_values = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split(",")
                if parts and parts[-1]:
                    try:
                        mse = float(parts[-1])
                        mse_values.append(mse)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"No se encontró el archivo: {log_file}")
    return mse_values

def plot_all_mse(mse_dict, title="MSE vs Épocas", save_as=None):
    """
    Grafica múltiples curvas de MSE vs Épocas.
    
    Args:
        mse_dict (dict): {label: [mse_values]}
        title (str): Título del gráfico.
        save_as (str): Ruta para guardar la imagen.
    """
    plt.figure(figsize=(10, 6))
    for label, mse_values in mse_dict.items():
        plt.plot(mse_values, label=label, linewidth=1.5)
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()

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

def main():
    lrs = [0.01, 0.001, 0.0001]
    _, epochs, epsilon, beta, input_file, _ = parse_params("params.json")
    X, y = parse_training_data(input_file)

    linear_mse_logs = {}
    nonlinear_mse_logs = {}

    for learn_rate in lrs:
        # Entrenamiento Linear
        perceptron = LinearPerceptron(learn_rate, 2000, epsilon)
        perceptron.train(X, y)
        mse_lin = read_mse_from_log("training_log_lin.txt")
        linear_mse_logs[f"Learning Rate={learn_rate}"] = mse_lin

        # Entrenamiento Non-Linear
        perceptron = NonLinearPerceptron(learn_rate, 30000, epsilon, beta)
        perceptron.train(X, y)
        mse_nonlin = read_mse_from_log("training_log_nonlin.txt")
        nonlinear_mse_logs[f"Learning Rate={learn_rate}"] = mse_nonlin

    # Graficar todos los lineales en un solo gráfico
    plot_all_mse(linear_mse_logs, title="[Lineal] MSE vs Épocas", save_as="all_linear.png")

    # Graficar todos los no lineales en otro gráfico
    plot_all_mse(nonlinear_mse_logs, title="[No Lineal] MSE vs Épocas", save_as="all_nonlinear.png")

if __name__ == "__main__":
    main()