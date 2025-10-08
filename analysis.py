import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Cargar el log
with open("training_log.txt", "r") as f:
    lines = f.readlines()

# Cargar pesos, bias y MSE por cada paso
history = [list(map(float, line.strip().split(","))) for line in lines]

# Separar en listas individuales
weights_bias = [entry[:3] for entry in history]
mses = [entry[3] for entry in history]

# Datos de entrenamiento (AND clásico en [-1, 1])
X = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])
y = np.array([-1, -1, -1, 1])

# Separar clases para graficar
pos = X[y == 1]
neg = X[y == -1]

# Crear la figura para la animación
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title("Evolución de la barrera de decisión")
ax.set_xlabel("x1")
ax.set_ylabel("x2")

# Puntos de datos
ax.scatter(pos[:, 0], pos[:, 1], color="blue", label="Clase +1")
ax.scatter(neg[:, 0], neg[:, 1], color="red", label="Clase -1")
line, = ax.plot([], [], 'k--', linewidth=2, label="Línea de decisión")

ax.legend()

# Función que actualiza la animación
def update(frame):
    w1, w2, b = weights_bias[frame]

    # Evitar división por cero
    if w2 == 0:
        w2 = 1e-6

    x_vals = np.array([-2, 2])
    y_vals = -(w1 * x_vals + b) / w2
    line.set_data(x_vals, y_vals)
    ax.set_title(f"Paso {frame + 1}")
    return line,

# Crear la animación
ani = animation.FuncAnimation(fig, update, frames=len(weights_bias), interval=1, repeat=False)

# Guardar como .gif
ani.save("decision_boundary.gif", writer="ffmpeg", fps=1)

# Crear gráfico de MSE vs Epoch
plt.figure()
plt.plot(range(1, len(mses) + 1), mses, marker='o', color='green')
plt.title("MSE vs Época")
plt.xlabel("Época")
plt.ylabel("Error cuadrático medio (MSE)")
plt.grid(True)
plt.savefig("mse_vs_epoch.png")
plt.show()
