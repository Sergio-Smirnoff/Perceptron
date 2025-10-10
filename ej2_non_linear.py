import numpy as np
import pandas as pd
import json
from NonLinearPerceptron import NonLinearPerceptron
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


def parse_params(params_path="params.json"):
    with open(params_path) as f:
        return json.load(f)


def parse_training_data(csv_file):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def test_perceptron(input_file, learn_rate, epochs, epsilon, beta, k_folds=7, seed=267, verbose=False):
    X_raw, y_raw = parse_training_data(input_file)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    mse_list = []
    fold_count = 1

    for train_idx, test_idx in kf.split(X_raw):
        print(f"\n=== Fold {fold_count}/{k_folds} ===")

        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]

        x_scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = x_scaler.fit_transform(X_train_raw)
        X_test_scaled = x_scaler.transform(X_test_raw)

        perc = NonLinearPerceptron(
            learning_rate=learn_rate,
            epochs=epochs,
            epsilon=epsilon,
            beta=beta,
            random_seed=seed
        )
        perc.train(X_train_scaled, y_train, verbose=False)

        preds = perc.predict(X_test_scaled)

        print("Input\t\tPredicted\tExpected")
        for i in range(min(4, len(X_test_raw))):
            print(f"{np.round(X_test_raw[i], 3)}\t{float(preds[i]):.4f}\t\t{float(y_test[i]):.4f}")

        mse = np.mean((preds - y_test) ** 2)
        mse_list.append(mse)
        print(f"MSE Fold {fold_count}: {mse:.6f}")
        fold_count += 1

    print("\n=== Resultados Finales ===")
    print(f"MSE promedio: {np.mean(mse_list):.6f}")
    print(f"MSE std: {np.std(mse_list):.6f}")
    return mse_list


if __name__ == "__main__":
    params = parse_params("ej2/params.json")
    learn_rate = float(params.get("learn_rate", 0.001))
    epochs = int(params.get("epochs", 2000))
    epsilon = float(params.get("epsilon", 1e-4))
    input_file = params.get("input_file", "ej2/TP3-ej2-conjunto.csv")
    beta = float(params.get("non_linear_beta", 1.0))

    test_perceptron(input_file, learn_rate, epochs, epsilon, beta)





