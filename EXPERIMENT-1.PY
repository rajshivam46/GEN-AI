import numpy as np
import matplotlib.pyplot as plt

# Normal Distribution
normal_data = np.random.normal(loc=0, scale=1, size=1000)

# Uniform Distribution
uniform_data = np.random.uniform(low=0, high=1, size=1000)

# Plot
plt.hist(normal_data, bins=30, alpha=0.5, label='Normal')
plt.hist(uniform_data, bins=30, alpha=0.5, label='Uniform')
plt.legend()
plt.show()
