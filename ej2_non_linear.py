import json
import numpy as np
import pandas as pd
from NonLinearPerceptron import NonLinearPerceptron

def parse_params(params):
    with open(params) as f:
        params = json.load(f)
    learn_rate = float(params.get("learn_rate", 0.01))
    epochs = int(params.get("epochs", 1000))
    epsilon = float(params.get("epsilon", 1e-6))
    input_file = params["input_file"]
    output_file = params.get("output_file", "out.csv")
    return learn_rate, epochs, epsilon, input_file, output_file

def parse_training_data(file_path):
    df = pd.read_csv(file_path)
    # última columna = salida (y)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=float)
    return X, y

def k_fold_indices(n_samples, k, seed=0):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)
    return folds

def test_perceptron(learn_rate, epochs, epsilon, X, y, beta=1.0, k_folds=10, seed=0, verbose=False):
    folds = k_fold_indices(len(X), k_folds, seed=seed)
    test_errors = []

    for i in range(k_folds):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k_folds) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Inicializar y entrenar el perceptrón
        perceptron = NonLinearPerceptron(
            learning_rate=learn_rate, epochs=epochs, epsilon=epsilon, beta=beta
        )
        perceptron.train(X_train, y_train, verbose=False)

        # Evaluar en entrenamiento
        train_preds = np.array([perceptron.predict(xi) for xi in X_train])
        train_mse = np.mean((y_train - train_preds) ** 2)
        print(f"Fold {i+1}/{k_folds} - MSE (train set): {train_mse:.6f}")

        # Evaluar en test
        preds = np.array([perceptron.predict(xi) for xi in X_test])
        mse = np.mean((preds - y_test) ** 2)
        test_errors.append(mse)

        # Mostrar algunos ejemplos
        print(f"\n=== Fold {i+1}/{k_folds} ===")
        print("Input\t\tPredicted\tExpected")
        for xt, yp, yr in zip(X_test[:5], preds[:5], y_test[:5]):
            print(f"{np.round(xt,3)}\t{float(yp):.4f}\t{float(yr):.4f}")
        print(f"MSE (test set): {mse:.6f}\n")

    avg_mse = np.mean(test_errors)
    print(f"\nPromedio de MSE en test (perceptrón): {avg_mse:.6f}")
    return avg_mse

def main():
    learn_rate, epochs, epsilon, input_file, output_file = parse_params("params.json")
    X, y = parse_training_data(input_file)
    avg_mse = test_perceptron(
        learn_rate, epochs, epsilon, X, y, beta=2.0, k_folds=10, seed=0
    )
    print(f"\nError cuadrático medio promedio: {avg_mse:.6f}")

if __name__ == "__main__":
    main()

