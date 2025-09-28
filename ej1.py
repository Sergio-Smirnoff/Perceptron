# ej1
from SimplePerceptron import SimplePerceptron

def main():
    training_set_and = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    expected_output_and = [-1, -1, -1, 1]
    training_set_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    expected_output_xor = [1, 1, -1, -1]
    perceptron = SimplePerceptron(0.01)

    perceptron.train(training_set_and, expected_output_and)
    input = [-1, 1]
    output = perceptron.predict(input)
    print(f"result for [-1, 1] after training with AND was {output}")
    input = [1, -1]
    output = perceptron.predict(input)
    print(f"result for [1, -1] after training with AND was {output}")
    input = [-1, -1]
    output = perceptron.predict(input)
    print(f"result for [-1, -1] after training with AND was {output}")
    input = [1, 1]
    output = perceptron.predict(input)
    print(f"result for [1, 1] after training with AND was {output}")

if __name__ == "__main__":
    main()