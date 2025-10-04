from NonLinearPerceptron import NonLinearPerceptron
import json
import pandas as pd
import numpy as np

def parse_params(params):
    with open(params) as f:
        params = json.load(f)
    learn_rate = params["learn_rate"]
    epochs = params["epochs"]
    epsilon = params["epsilon"]
    input_file = params["input_file"]
    output_file = params["output_file"]
    return learn_rate, epochs, epsilon, input_file, output_file


def parse_training_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y


def split_dataset(X, y, test_ratio=0.2):
    total_samples = len(X)
    test_size = int(total_samples * test_ratio)

    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test


def test_perceptron(learn_rate, epochs, epsilon, X, y, beta, k_folds=10):
    np.random.seed(0)
    test_errors = []

    for fold in range(k_folds):
        # Dividir dataset
        X_train, y_train, X_test, y_test = split_dataset(X, y)
        perceptron = NonLinearPerceptron(learn_rate, epochs, epsilon, beta)
        perceptron.train(X_train, y_train)

        print(f"\n=== Fold {fold+1}/{k_folds} ===")
        print("Input\t\tPredicted\tExpected")
        preds = []
        for x_t, y_real in zip(X_test, y_test):
            y_pred = perceptron.predict(x_t)
            preds.append(y_pred)
            print(f"{np.round(x_t,3)}\t{y_pred:.4f}\t{y_real:.4f}")

        preds = np.array(preds)
        mse = np.mean((preds - y_test)**2)
        test_errors.append(mse)
        print(f"MSE (test set): {mse:.6f}")

    avg_mse = np.mean(test_errors)
    print(f"\nPromedio de MSE en test: {avg_mse:.6f}")
    return avg_mse


def main():
    learn_rate, epochs, epsilon, input_file, output_file = parse_params("params.json")
    X, y = parse_training_data(input_file)

    avg_mse = test_perceptron(learn_rate, epochs, epsilon, X, y, beta=0.5, k_folds=100)
    print(f"Error cuadrático medio promedio: {avg_mse:.6f}")


if __name__ == "__main__":
    main()
