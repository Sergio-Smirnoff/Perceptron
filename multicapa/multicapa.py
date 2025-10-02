
import logging as log
import SimplePerceptron as sp
import numpy as np
import random
from tqdm import tqdm

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MulticapaXOR:

    def __init__(self, learning_rate:float, epochs:int=100, epsilon:float=0.0):
        self.capas = 2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.trainings = {
        "AND": 
            { 
                "X":[[-1, 1], [1, -1], [-1, -1], [1, 1]],
                "z": [-1, -1, -1, 1]
            },
        "NAND":
            {
                "X": [[-1, 1], [1, -1], [-1, -1], [1, 1]],
                "z": [1, 1, 1, -1]
            },
        "OR":
            {
                "X": [[-1, 1], [1, -1], [-1, -1], [1, 1]],
                "z": [1, 1, -1, 1]
            },
        }
        self.perceptrons = [
            [
                sp.SimplePerceptron(learning_rate, epochs, epsilon),
                sp.SimplePerceptron(learning_rate, epochs, epsilon)
            ],
            [
                sp.SimplePerceptron(learning_rate, epochs, epsilon)
            ]
        ]

    # revisar
    def _step_activation_function(self, x: float) -> int:
        return 1 if x >= 0 else -1

    def train_perceptrons(self):
        log.info("Entrenando capas...")
        self.perceptrons[1][0].train(self.trainings["AND"]["X"], self.trainings["AND"]["z"])
        self.perceptrons[0][1].train(self.trainings["NAND"]["X"], self.trainings["NAND"]["z"])
        self.perceptrons[0][0].train(self.trainings["OR"]["X"], self.trainings["OR"]["z"])
        log.info("Capas entrenadas.")


    def train(self, X:np.ndarray, z:np.ndarray):
        log_file = open("training_log.txt", "w")  
        ## revisar tema de pesos 
        self.weights = np.array([random.uniform(-0.5, 0.5) for _ in range(len(X[0]))])
        self.bias = random.uniform(-0.5, 0.5)
        self.train_perceptrons()

        ## stuff


        log_file.close()
        log.info("Entrenamiento finalizado.")

    def predict(self, x:np.ndarray) -> int:
        dads()