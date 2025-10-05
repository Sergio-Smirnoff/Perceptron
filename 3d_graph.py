import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from tqdm import tqdm
import os

# === 1. Load 3D dataset ===
# Assuming CSV format: x1,x2,x3,label
points = np.loadtxt("../TP3-ej2-conjunto.csv", delimiter=",", skiprows=1)
X = points[:, :3]   # first 3 columns are features
if points.shape[1] > 3:
    labels = points[:, 3]
    colors = ['r' if l == 0 else 'b' for l in labels]
else:
    labels = None
    colors = 'k'

# === 2. Load perceptron training log ===
weights_bias_list = []
with open("training_log.txt", "r") as f:
    for line in f:
        vals = [float(x) for x in line.strip().split(",")]
        weights_bias_list.append(vals)
weights_bias_array = np.array(weights_bias_list)

# === 3. Set up ranges for plane plotting ===
x_range = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 20)
y_range = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 20)
Xgrid, Ygrid = np.meshgrid(x_range, y_range)

# Fix axes limits for all frames
xlim = (X[:,0].min()-0.5, X[:,0].max()+0.5)
ylim = (X[:,1].min()-0.5, X[:,1].max()+0.5)
zlim = (X[:,2].min()-0.5, X[:,2].max()+0.5)

# === 4. Create frames ===
frames = []
tmp_files = []
for step, wb in enumerate(tqdm(weights_bias_array, desc="Creating frames")):
    w1, w2, w3, b = wb

    if abs(w3) > 1e-6:
        Zgrid = (-w1*Xgrid - w2*Ygrid - b) / w3
    else:
        Zgrid = np.zeros_like(Xgrid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=colors, s=40, depthshade=True)
    ax.plot_surface(Xgrid, Ygrid, Zgrid, alpha=0.5, color='cyan')

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")
    ax.set_title(f"Perceptron Step {step+1}")

    # Keep fixed axis ranges
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    plt.tight_layout()
    os.makedirs("frames", exist_ok=True)
    fname = f"frames/_frame_{step:03d}.png"
    plt.savefig(fname)
    tmp_files.append(fname)
    frames.append(imageio.imread(fname))
    plt.close()

# === 5. Save GIF ===
gif_filename = "perceptron_training_3d_(e0.01-2).gif"
imageio.mimsave(gif_filename, frames, fps=2)
print(f"GIF saved as {gif_filename}")

# === 6. Cleanup ===
for f in tmp_files:
    os.remove(f)
