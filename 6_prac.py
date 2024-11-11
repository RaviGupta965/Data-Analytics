import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the exponential distribution
def exp_dist(x, lam):
    prob_density = lam * np.exp(-lam * x)
    return prob_density

# Generate and plot the exponential distribution for various values of Lambda
def plot_exp_dist(lambdas, x_range):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration
    
    for i, lam in enumerate(lambdas):
        x = np.arange(0, x_range, 0.1)
        result = exp_dist(x, lam)
        
        axs[i].scatter(x, result)
        axs[i].set_title(f'λ={lam}')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('Probability Density')
    
    # Adjust layout to prevent overlap
    fig.tight_layout()
    plt.show()

# List of Lambda values to demonstrate
lambdas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
x_range = 10  # Range of x values

# Call the function to generate and plot the exponential distribution
plot_exp_dist(lambdas, x_range)
