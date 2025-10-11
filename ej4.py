import numpy as np
from keras.datasets import mnist

# Hace un arreglo con la clasificacion de numeros, entonces si nececito 10 clases para los numeros de 0 a 9,
# va a ser ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]...)
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

# Va a calcular las probabilidad de que la imágen sea de N numero entre 0 a 9 previo a pasarselo a la capa clasificadora
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Convierte los numeros negativos en 0
def relu(z):
    return np.maximum(0, z)

# Retorna 1 si es positivo
def relu_derivative(z):
    return (z > 0).astype(float)

def cross_entropy(y_pred, y_true):
    m = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.sum(y_true * np.log(y_pred)) / m

class MLP:
    def __init__(self, input_size=784, hidden1=128, hidden2=64, output_size=10, lr=0.01):
        self.lr = lr

        # Inicialización de pesos
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(2. / hidden2)
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = relu(self.Z2)
        self.Z3 = self.A2.dot(self.W3) + self.b3
        self.A3 = softmax(self.Z3)
        return self.A3

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        dZ3 = y_pred - y_true
        dW3 = (1/m) * self.A2.T.dot(dZ3)
        db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3.dot(self.W3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = (1/m) * self.A1.T.dot(dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (1/m) * X.T.dot(dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Actualización de pesos
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=10, batch_size=64):
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]
            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred)

            y_pred_full = self.forward(X)
            loss = cross_entropy(y_pred_full, y)
            acc = np.mean(np.argmax(y_pred_full, axis=1) == np.argmax(y, axis=1))
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0
y_train = one_hot(y_train)
y_test = one_hot(y_test)

mlp = MLP(lr=0.01)
mlp.train(X_train[:20000], y_train[:20000], epochs=100)  # usa un subset para hacerlo rápido

y_pred = mlp.forward(X_test)
test_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
print(f"Test accuracy: {test_acc*100:.2f}%")

indices = np.random.choice(len(X_test), 500, replace=False)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nEjemplos de clasificación:")
print("=" * 40)
for i in indices:
    print(f"Sample {i:5d} | Expected: {y_true_classes[i]} | Predicted: {y_pred_classes[i]}")

