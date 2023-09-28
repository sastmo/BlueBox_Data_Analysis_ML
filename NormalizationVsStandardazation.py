# https://chat.openai.com/share/d1dbb828-c41c-40a9-b067-1825d92ce482

# Standardization Example:
import numpy as np

# Sample data
data = np.array([13, 16, 19, 22, 25])

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Standardize the data
standardized_data = (data - mean) / std_dev

print("Original Data:", data)
print("Standardized Data:", standardized_data)

print("_________________________________________________")

# Normalization Example:
import numpy as np

# Sample data
data = np.array([13, 16, 19, 22, 25])

# Calculate minimum and maximum values
min_value = np.min(data)
max_value = np.max(data)

# Normalize the data
normalized_data = (data - min_value) / (max_value - min_value)

print("Original Data:", data)
print("Normalized Data:", normalized_data)
