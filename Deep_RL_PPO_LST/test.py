import numpy as np
import matplotlib.pyplot as plt

# Constants
portfolio = 1000
risk = 0.02
size = 100000
fees = 0.0005

# Range for the "value" parameter from -0.02 to 1
value_range = np.linspace(-0.02, 0.01, 500)

# Lists to store results
delta_size_values = []
percentage_values = []

# Calculate delta_size and percentage for each value in the range
for value in value_range:
    delta_size = (2 * size * fees - risk * portfolio - risk * value * portfolio) / (
        2 * fees
    )
    percentage = (value * portfolio * 2 * fees) / (
        2 * size * fees - risk * portfolio - risk * value * portfolio
    )

    delta_size_values.append(delta_size)
    percentage_values.append(percentage)

# Plotting delta_size and percentage against value
plt.figure(figsize=(12, 6))

# Plot for delta_size
plt.subplot(1, 2, 1)
plt.plot(value_range, delta_size_values, label="Delta Size", color="blue")
plt.title("Delta Size vs Value")
plt.xlabel("Value")
plt.ylabel("Delta Size")
plt.grid(True)

# Plot for percentage
plt.subplot(1, 2, 2)
plt.plot(value_range, percentage_values, label="Percentage Change", color="green")
plt.title("Percentage Change vs Value")
plt.xlabel("Value")
plt.ylabel("Percentage Change")
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
