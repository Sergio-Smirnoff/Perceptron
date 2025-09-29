
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

# === PARAMETERS ===
csv_file = "../linear_test.csv"       # CSV with columns x and y
log_file = "training_log.txt"  # log.txt with weight1,weight2,...,bias format
gif_file = "../perceptron_training.gif"
fps = 5                     # frames per second for the GIF

# === 1. Load data ===
df = pd.read_csv(csv_file)
X = df.iloc[:, 0].to_numpy()  # assume first column is x
Y = df.iloc[:, 1].to_numpy()  # second column is y

# === 2. Load log.txt ===
log_lines = []
with open(log_file, "r") as f:
    for line in f:
        # split by commas and convert to float
        values = [float(v) for v in line.strip().split(",")]
        log_lines.append(values)

# Separate weights and bias for 2D case (linear function y = w*x + b)
# Assumes single weight for x plus bias
frames = []
for step_idx, step_values in tqdm(enumerate(log_lines), desc="Creating frames", total=len(log_lines)):
    weight = step_values[0]
    bias = step_values[-1]

    # Plot data points
    plt.figure(figsize=(6,4))
    plt.scatter(X, Y, color='blue', label='Data')
    
    # Plot perceptron line with step in the label
    x_vals = np.linspace(min(X)-1, max(X)+1, 200)
    y_vals = weight * x_vals + bias
    plt.plot(x_vals, y_vals, color='red', 
             label=f'y = {weight:.2f}x + {bias:.2f}\n\n(Step {step_idx+1})')
    
    plt.title("Linear Perceptron Training")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.xlim(min(X)-1, max(X)+1)
    plt.ylim(min(Y)-1, max(Y)+1)
    
    # Save figure to image in memory
    plt.tight_layout()
    plt.savefig("temp_frame.png")
    plt.close()
    frames.append(imageio.imread("temp_frame.png"))

# === 3. Save animated GIF ===
imageio.mimsave(gif_file, frames, fps=fps)
print(f"GIF saved as {gif_file}")
