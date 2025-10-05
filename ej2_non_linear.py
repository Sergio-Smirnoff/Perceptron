# ej2_non_linear.py
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
    # last column assumed y
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=float)
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

    for i in range(k_folds):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k_folds) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Initialize and train perceptron
        perceptron = NonLinearPerceptron(learning_rate=learn_rate, epochs=epochs, epsilon=epsilon, beta=beta)
        perceptron.train(X_train, y_train, verbose=False)

        # Predict (vectorized)
        preds = perceptron.predict(X_test)  # returns 1D array
        mse = np.mean((preds - y_test)**2)
        test_errors.append(mse)

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

def main():
    learn_rate, epochs, epsilon, input_file, output_file = parse_params("params.json")
    X, y = parse_training_data(input_file)
    avg_mse, avg_baseline = test_perceptron(learn_rate, epochs, epsilon, X, y, beta=2.0, k_folds=10, seed=0)
    print(f"\nError cuadrático medio promedio: {avg_mse:.6f}")
    print(f"Baseline lineal promedio: {avg_baseline:.6f}")

if __name__ == "__main__":
    main()

