import numpy as np
from LinearPerceptron import LinearPerceptron
import json
import pandas as pd
from tqdm import tqdm
from NonLinearPerceptron import NonLinearPerceptron
from ClassifierPerceptron import ClassifierPerceptron
import matplotlib.pyplot as plt
import shutil
import os

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


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

def k_fold_indices(n_samples, k, seed=0):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)
    return folds

def linear_regression_baseline(X_train, y_train, X_test):
    # solve on raw X_train but better to scale X to [0,1] using train's min/max
    # here expects X already scaled (in [0,1]) and y scaled to [0,1]
    # We'll implement in caller correctly.
    Xb = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    w, *_ = np.linalg.lstsq(Xb, y_train, rcond=None)
    Xb_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    y_pred = Xb_test.dot(w)
    return y_pred

def test_perceptron(learn_rate, epochs, epsilon, X, y, beta=1.0, k_folds=10, seed=0, verbose=False):
    folds = k_fold_indices(len(X), k_folds, seed=seed)
    test_errors = []
    baseline_errors = []
    train_histories = []

    for i in range(k_folds):
        open("test_log_nonlin.txt", "w").close()
        open("training_log_nonlin.txt", "w").close()
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k_folds) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Initialize and train perceptron
        perceptron = NonLinearPerceptron(learning_rate=learn_rate, epochs=1, epsilon=epsilon, beta=beta)
        for _ in range(epochs):
            perceptron.train(X_train, y_train, verbose=False)
            train_histories.append(perceptron.history)

            # Predict (vectorized)
            preds = perceptron.predict(X_test)  # returns 1D array
            mse = np.mean((preds - y_test)**2)
            test_errors.append(mse)
            with open("test_log_nonlin.txt", "a") as f_test:
                f_test.write(f"{mse}\n")
        
        os.makedirs("logs", exist_ok=True)
        shutil.copy("training_log_nonlin.txt", f"logs/fold_{i}_log.txt")
        shutil.copy("test_log_nonlin.txt", f"logs/fold_{i}_test_log.txt")

        # Baseline: linear regression on scaled training data using the same scaler as perceptron
        # we must use the perceptron's internal scaler to scale X_train and X_test and y_train
        # Build scaled arrays:
        denom_X = (perceptron.X_max - perceptron.X_min)
        denom_X[denom_X == 0] = 1e-9
        X_train_scaled = (X_train - perceptron.X_min) / denom_X
        X_test_scaled = (X_test - perceptron.X_min) / denom_X

        denom_y = (perceptron.y_max - perceptron.y_min) if (perceptron.y_max - perceptron.y_min) != 0 else 1e-9
        y_train_scaled = (y_train - perceptron.y_min) / denom_y

        baseline_scaled_pred = linear_regression_baseline(X_train_scaled, y_train_scaled, X_test_scaled)
        baseline_pred = baseline_scaled_pred * denom_y + perceptron.y_min
        baseline_mse = np.mean((baseline_pred - y_test)**2)
        baseline_errors.append(baseline_mse)

        # print few examples
        print(f"\n=== Fold {i+1}/{k_folds} ===")
        print("Input\t\tPredicted\tExpected")
        for xt, yp, yr in zip(X_test[:5], preds[:5], y_test[:5]):
            print(f"{np.round(xt,3)}\t{float(yp):.4f}\t{float(yr):.4f}")
        print(f"MSE (test set): {mse:.6f}  | baseline linear MSE: {baseline_mse:.6f}")


    avg_mse = np.mean(test_errors)
    avg_baseline = np.mean(baseline_errors)
    print(f"\nPromedio de MSE en test (perceptrón): {avg_mse:.6f}")
    print(f"Promedio de MSE baseline lineal: {avg_baseline:.6f}")
    return avg_mse, avg_baseline

def pad_curves(curves, target_len):
    padded = []
    for curve in curves:
        if len(curve) < target_len:
            pad_len = target_len - len(curve)
            curve = curve + [np.nan] * pad_len
        padded.append(curve)
    return padded

def main():
    k = 3 #[2, 4, 7, 14]
    learn_rate, epochs, epsilon, beta, input_file, _ = parse_params("params.json")
    X, y = parse_training_data(input_file)

    test_perceptron(learn_rate, epochs, epsilon, X, y, beta=beta, k_folds=k, seed=0)
    # Calcular promedio del MSE por época
    train_curves = []
    test_curves = []

    for i in range(k):
        train_path = f"logs/fold_{i}_log.txt"
        test_path = f"logs/fold_{i}_test_log.txt"

        # Leer mse de entrenamiento (último valor de cada línea)
        with open(train_path) as f:
            train_mse = [float(line.strip().split(',')[-1]) for line in f]
            train_curves.append(train_mse)

        # Leer mse de test
        with open(test_path) as f:
            test_mse = [float(line.strip()) for line in f]
            test_curves.append(test_mse)

    # === Promedios ignorando NaNs ===
    max_len = max(max(len(c) for c in train_curves), max(len(c) for c in test_curves))
    train_curves = pad_curves(train_curves, max_len)
    test_curves = pad_curves(test_curves, max_len)

    avg_train = np.nanmean(train_curves, axis=0)
    avg_test = np.nanmean(test_curves, axis=0)

    # Graficar
    plt.plot(avg_train, label=f'train [k={k}]')
    plt.plot(avg_test, label=f'test [k={k}]')

    plt.xlabel('Época')
    plt.ylabel('MSE entrenamiento')
    plt.title('Promedio de MSE por época')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gen_mse.png")
    plt.show()


if __name__ == "__main__":
    main()