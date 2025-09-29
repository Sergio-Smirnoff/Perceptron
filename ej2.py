from LinearPerceptron import LinearPerceptron
import json
import pandas as pd
def parse_params(params):
    with open(params) as f:
        params = json.load(f)
    learn_rate = params["learn_rate"]
    epochs = params["epochs"]
    epsilon = params["epsilon"]
    return learn_rate, epochs, epsilon

#====Parse CSV Training Data====
def parse_training_data(file_path):
    df = pd.read_csv(file_path)
    # Each row is a sample with all its features
    X = df[['x1', 'x2', 'x3']].values.tolist()  
    y = df['y'].tolist()
    return X, y

def main():
    learn_rate, epochs, epsilon = parse_params("../params.json")
    X, y = parse_training_data("../TP3-ej2-conjunto.csv")
    perceptron = LinearPerceptron(learn_rate, epochs, epsilon)
    perceptron.train(X, y)


if __name__ == "__main__":
    main()