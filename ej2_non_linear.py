from NonLinearPerceptron import NonLinearPerceptron
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

def parse_params(params):
    with open(params) as f:
        params = json.load(f)
    learn_rate = params["learn_rate"]
    epochs = params["epochs"]
    epsilon = params["epsilon"]
    input_file = params["input_file"]
    output_file = params["output_file"]
    return learn_rate, epochs, epsilon, input_file, output_file

#====Parse CSV Training Data====
def parse_training_data(file_path):
    df = pd.read_csv(file_path)

    # Each row is a variable -> support for multivalued functions
    X = df.iloc[:, :-1].to_numpy(dtype=float)

    #Last column is expected value
    y = df['y'].to_numpy(dtype=float)
    return X, y

#======= Divide dataset into training and testing sets =======
def split_dataset(X, y, test_ratio=0.5):
    """Split dataset into training and testing sets.
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        test_ratio (float): Proportion of the dataset to include in the test split.
    """
    total_samples = len(X)
    test_size = int(total_samples * test_ratio)

    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test


def main():
    learn_rate, epochs, epsilon, input_file, output_file = parse_params("../params.json")
    X, y = parse_training_data("../" + input_file)

    y_pred = test_perceptron(learn_rate, epochs, epsilon, X, y)

    

def test_perceptron(learn_rate, epochs, epsilon, X: np.ndarray, y: np.ndarray):
    """Predict output. Generate a new perceptron for each new split of the dataset.
    """
    mean_acc = 0.0
    for _ in range(10):  # Number of iterations to average accuracy
        X_train, y_train, X_test, y_test = split_dataset(X, y)
        perceptron = NonLinearPerceptron(learn_rate, epochs, epsilon)
        perceptron.train(X_train, y_train)
        for x_test, y_test in zip(X_test, y_test):
            y_pred = perceptron.predict(x_test)
            print(f"Input: {x_test}, Predicted: {y_pred}, Expected: {y_test}")
    return y_pred

if __name__ == "__main__":
    main()