import numpy as np
from LinearPerceptron import LinearPerceptron
import json
import pandas as pd

from NonLinearPerceptron import NonLinearPerceptron

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
def split_dataset(X, y, test_ratio=0.2):
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

def get_best_T(X, y):
    perceptron = LinearPerceptron(learning_rate=0.01, epochs=100, epsilon=0.01)
    best_T = None
    best_accuracy = 0

    for i in range(10):
        X_train, y_train, X_test, y_test = split_dataset(X, y) #try different splits of the dataset
        for T in np.arange(-10, 10, 0.01):
            perceptron.set_threshold(T)
            perceptron.train(X_train, y_train)
            y_pred = perceptron.predict(X_test)
            accuracy = np.mean(y_pred == y_test)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_T = T
    print
    return best_T

def main():
    learn_rate, epochs, epsilon, input_file, output_file = parse_params("params.json")
    X, y = parse_training_data(input_file)
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    perceptron = LinearPerceptron(learn_rate, epochs, epsilon)
    perceptron.train(X, y)

    #una vez entrenado el perceptron, se obtiene un hiperplano de la forma X.W+b=0
    #donde X es el vector de inputs, W es el vector de pesos y b es el bias

    #Luego hace falta calibrar el umbral de clasificacion T, que sera el valor que separa las dos clases
    # X.W + b > 0 -> clase 1
    # X.W + b <= 0 -> clase 0

    # f(x1, x2, x3)=X -> X1 + X2 + X3

    for i in range(0.0, 50, 0.01):
        perceptron.predict(X_test)


if __name__ == "__main__":
    main()

def predict(dataset: str, perceptron: LinearPerceptron):
    test_inputs = [2, 4, 6, 8, 54, 100, 123, 256, 512, 1024]
    for x in test_inputs:
        output = perceptron.predict(np.array(x))
        print(f"result for {x} after training with f(x)=x was {output}")

def predict(dataset: str, perceptron: NonLinearPerceptron):
    test_inputs = [2, 4, 6, 8, 54, 100, 123, 256, 512, 1024]
    for x in test_inputs:
        output = perceptron.predict(np.array(x))
        print(f"result for {x} after training with f(x)=x^2 was {output}")
    for x in test_inputs:
        output = perceptron.predict(np.array(x))
        print(f"result for {x} after training with f(x)=x was {output}")