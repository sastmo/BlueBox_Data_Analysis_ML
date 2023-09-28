from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Build the time series, just a simple AR(1)
t1 = [0.1 * np.random.normal()]
for _ in range(100):
    t1.append(0.5 * t1[-1] + 0.1 * np.random.normal())

# Build the time series that is granger caused by t1
t2 = [item + 0.1 * np.random.normal() for item in t1]

# Adjust t1 and t2 to account for the initial values
t1 = t1[3:]
t2 = t2[:-3]

# Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(t1, color='b')
plt.plot(t2, color='r')
plt.legend(['t1', 't2'], fontsize=16)
plt.show()

# Create a DataFrame from the time series
ts_df = pd.DataFrame({'t2': t2, 't1': t1})

# Display the DataFrame
print(ts_df)

# Perform Granger causality tests with maximum lag of 3
max_lag = 3
gc_res = grangercausalitytests(ts_df, max_lag)

# Print the results of Granger causality tests
for lag in range(1, max_lag + 1):
    print(f"Results for lag {lag}:")
    print(gc_res[lag][0])  # F-test statistic, p-value, degrees of freedom
    print("===")
