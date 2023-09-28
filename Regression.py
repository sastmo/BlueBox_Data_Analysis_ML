# https://www.scribbr.com/statistics/statistical-tests/
# https://datatab.net/statistics-calculator/regression

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import f

# Here's the data
mouse_data = pd.DataFrame({
    'size': [1.4, 2.6, 1.0, 3.7, 5.5, 3.2, 3.0, 4.9, 6.3],
    'weight': [0.9, 1.8, 2.4, 3.5, 3.9, 4.4, 5.1, 5.6, 6.3],
    'tail': [0.7, 1.3, 0.7, 2.0, 3.6, 3.0, 2.9, 3.9, 4.0]
})

# Let's start by reviewing simple regression by modeling mouse size with mouse weight.

# STEP 1: Draw a graph of the data to make sure the relationship makes sense
plt.scatter(mouse_data['weight'], mouse_data['size'], marker='o')
plt.xlabel('Weight')
plt.ylabel('Size')
plt.title('Size vs Weight')
plt.show()

# STEP 2: Do the regression
X_simple = sm.add_constant(mouse_data['weight'])  # Add a constant term for the intercept
simple_regression = sm.OLS(mouse_data['size'], X_simple).fit()

# STEP 3: Look at the R^2, F-value, and p-value
print(simple_regression.summary())

# Now let's do multiple regression by adding an extra term, tail length

# STEP 1: Create a 3x3 grid layout for the scatter plots
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# STEP 2: Do the regression and plot for each pair of variables
for i, x_var in enumerate(mouse_data.columns):
    for j, y_var in enumerate(mouse_data.columns):
        X = sm.add_constant(mouse_data[[x_var]])  # Add a constant term for the intercept
        regression_model = sm.OLS(mouse_data[y_var], X).fit()
        axes[i, j].scatter(mouse_data[x_var], mouse_data[y_var], marker='o')
        axes[i, j].set_xlabel(x_var)
        axes[i, j].set_ylabel(y_var)
        axes[i, j].set_title(f"{y_var} vs {x_var}\nR-squared: {regression_model.rsquared:.3f}")
        axes[i, j].plot(mouse_data[x_var], regression_model.predict(X), color='red')

# Calculate multiple regression
X_multiple = sm.add_constant(mouse_data[['weight', 'tail']])  # Add a constant term for the intercept
multiple_regression = sm.OLS(mouse_data['size'], X_multiple).fit()

# Print the summary for multiple regression
print(multiple_regression.summary())

plt.show()

# Now, let's see if "tail" makes a significant contribution by comparing the "simple" fit
# to the "multiple" fit.

# Calculate F-value for the comparison
f_simple_vs_multiple = ((simple_regression.ssr - multiple_regression.ssr) / (2 - 1)) / (multiple_regression.ssr / (mouse_data.shape[0] - 2))

# Get the p-value for the F-value
p_value_simple_vs_multiple = 1 - f.cdf(f_simple_vs_multiple, 1, mouse_data.shape[0] - 2)

print(f"Comparison between simple and multiple regression: F-value = {f_simple_vs_multiple}, p-value = {p_value_simple_vs_multiple}")
