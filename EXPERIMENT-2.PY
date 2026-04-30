import numpy as np

# Generate data
data = np.random.normal(5, 2, 1000)

# MLE estimation
mu_est = np.mean(data)
sigma_est = np.var(data)

print("Estimated Mean:", mu_est)
print("Estimated Variance:", sigma_est)
