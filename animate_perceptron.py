import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Cargar el log
with open("training_log.txt", "r") as f:
    lines = f.readlines()

# Cargar pesos y bias por cada paso
history = [list(map(float, line.strip().split(","))) for line in lines]

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

# Crear la figura
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
    w1, w2, b = history[frame]

    # Evitar división por cero
    if w2 == 0:
        w2 = 1e-6

    x_vals = np.array([-2, 2])
    y_vals = -(w1 * x_vals + b) / w2
    line.set_data(x_vals, y_vals)
    ax.set_title(f"Paso {frame + 1}")
    print(f"Ordenada al origen (x1=0): {-b/w2}")
    print(f"Bias reportado: {b}")
    return line,

# Crear la animación
ani = animation.FuncAnimation(fig, update, frames=len(history), interval=50, repeat=False)

# Guardar como .mp4 o .gif si quieres
# ani.save("decision_boundary.mp4", writer="ffmpeg", fps=30)

plt.show()
