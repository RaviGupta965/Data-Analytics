import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# Function to calculate the binomial distribution
def binomial_dist(n, p, x):
    # Calculate the binomial coefficient(n choose x)
    binom_coeff = factorial(n)/(factorial(x)*factorial(n-x))
    # Calculate the PMF
    pmf = binom_coeff*(p**x)*((1-p))**(n-x)
    return pmf

# Values of n and p for the plots
n_values = [20, 40, 60]
p_values = [0.2, 0.5, 0.8]

# Create a figure with subplots
fig, axs = plt.subplots(len(n_values), len(p_values), figsize=(15, 12))

# Generate the plots for each combination of n and p
for i in range(len(n_values)):
    for j in range(len(p_values)):
        n = n_values[i]
        p = p_values[j]
        x = np.arange(0, n+1)
        result = binomial_dist(n, p, x)
        
        axs[i, j].bar(x, result, color='blue')
        axs[i, j].set_title(f'n={n}, p={p}')
        axs[i, j].set_xlabel('Number of Successes')
        axs[i, j].set_ylabel('Probability')

# Adjust the layout
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.4, wspace=0.3)

# Show the plots
plt.show()