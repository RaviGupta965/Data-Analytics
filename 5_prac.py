import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the normal distribution PDF
def normal_dist(x, mean, sd):
    prob_density = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density

# Define a list of means and standard deviations to demonstrate
means = [0, 1, -1]
sds = [1, 2, 0.5]

# Create subplots
fig, axs = plt.subplots(len(means), len(sds), figsize=(15, 10))

# Generate and plot the normal distribution PDF for each combination of mean and standard deviation
for i, mean in enumerate(means):
    for j, sd in enumerate(sds):
        x = np.linspace(mean - 3*sd, mean + 3*sd, 100)
        y = normal_dist(x, mean, sd)
        axs[i, j].plot(x, y)
        axs[i, j].set_title(f'µ={mean}, σ={sd}')
        axs[i, j].grid(True)

# Add a legend, title, and labels
for ax in axs.flat:
    ax.set(xlabel='x', ylabel='Probability Density')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.tight_layout()
plt.show()
