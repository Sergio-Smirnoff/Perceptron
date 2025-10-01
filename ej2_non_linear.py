from NonLinearPerceptron import NonLinearPerceptron
import json
import pandas as pd

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

def main():
    learn_rate, epochs, epsilon, input_file, output_file = parse_params("../params.json")
    X, y = parse_training_data("../" + input_file)
    perceptron = NonLinearPerceptron(learn_rate, epochs, epsilon)
    perceptron.train(X, y)

    test_inputs = [2, 4, 6, 8, 54, 100, 123, 256, 512, 1024]
    for x in test_inputs:
        output = perceptron.predict(np.array([x]))
        #print(f"result for {input_vec} after training with AND was {output}")
        print(f"result for {x} after training with XOR was {output}")

if __name__ == "__main__":
    main()