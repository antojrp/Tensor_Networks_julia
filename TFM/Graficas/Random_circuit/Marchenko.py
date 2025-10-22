

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Marchenko-Pastur distribution
sigma2 = 1  # Variance of the underlying distribution
lambda_ = 0.5  # Ratio n/m

# Compute support limits
a = sigma2 * (1 - np.sqrt(lambda_))**2
b = sigma2 * (1 + np.sqrt(lambda_))**2

# Define the Marchenko-Pastur density function
def marchenko_pastur(x, sigma2, lambda_):
    if a <= x <= b:
        return (1 / (2 * np.pi * lambda_ * sigma2)) * np.sqrt((b - x) * (x - a)) / x
    else:
        return 0

# Generate x values for plotting
x_vals = np.linspace(a, b, 400)
y_vals = [marchenko_pastur(x, sigma2, lambda_) for x in x_vals]

# Plot the Marchenko-Pastur distribution
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label=f"Marchenko-Pastur (λ={lambda_})", color='b')
plt.fill_between(x_vals, y_vals, alpha=0.3, color='b')
plt.xlabel("Eigenvalue λ")
plt.ylabel("Density")
plt.title("Marchenko-Pastur Distribution of Eigenvalues")
plt.legend()
plt.grid(True)
plt.show()