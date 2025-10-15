import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import logging as log


from multicapa.multiLayer import MultiLayer

EPOCHS = 7000

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

mlp = MultiLayer(
    layers_array=[784, 392, 196, 64, 10],
    learning_rate=0.01,
    epochs=1,
    epsilon=1e-4,
    optimization_mode="adam",
    loss_function="cross_entropy",
    seed=42
)

error_entropy_list = []
error_mse_list = []
epochs = []
log.info("Entrenando red neuronal...")
for epoch in range(EPOCHS):
    error_entropy, error_mse = mlp.train(X_train, y_train)
    print(f"Época {epoch + 1}/{EPOCHS} - Error Entropía: {error_entropy:.6f}, Error MSE: {error_mse:.6f}")
    error_entropy_list.append(error_entropy)
    error_mse_list.append(error_mse)
    epochs.append(epoch)
    if error_entropy < 1e-4:
        break


print(f"\nEntrenamiento finalizado.")
print(f"Error final (Cross-Entropy): {error_entropy:.6f}")
print(f"Error final (MSE): {error_mse:.6f}")

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy sobre test: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred, labels=np.arange(10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap="Blues", colorbar=True)
plt.title("Matriz de confusión - MLP MNIST")
plt.show()

plt.figure()
plt.plot(range(EPOCHS), np.full(EPOCHS, error_entropy_list), label="Error")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.title("Curva de Error")
plt.legend()
plt.show()

predictions = y_pred
correct_indices = np.where(predictions == y_test)[0]
incorrect_indices = np.where(predictions != y_test)[0]

plt.figure(figsize=(10, 4))
for i, idx in enumerate(correct_indices[:5]):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {predictions[idx]}")
    plt.axis("off")

for i, idx in enumerate(incorrect_indices[:5]):
    plt.subplot(2, 5, i+6)
    plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {predictions[idx]}\nGT: {y_test[idx]}")
    plt.axis("off")

plt.suptitle("Predicciones correctas (arriba) e incorrectas (abajo)")
plt.tight_layout()
plt.show()
