# https://github.com/ritvikmath/Time-Series-Analysis/blob/master/Undo%20Stationary%20Transformations.ipynb
# https://towardsdatascience.com/advanced-time-series-analysis-with-arma-and-arima-a7d9b589ed6d

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA  # Updated import


# Define a function to perform the Augmented Dickey-Fuller test
def perform_adf_test(series):
    """
    Perform the Augmented Dickey-Fuller (ADF) test on a given time series.

    Args:
        series (pd.Series): The time series to be tested.

    Returns:
        None: Prints ADF statistic and p-value.
    """
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])


# Load the original time series data
ts = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\original_series.csv")
ts.index = np.arange(1, len(ts) + 1)
# print(ts)

# Plot the original time series
plt.figure(figsize=(10, 4))
plt.plot(ts)
plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours Since Published', fontsize=16)
plt.yticks(np.arange(0, 50000, 10000), fontsize=14)
plt.ylabel('Views', fontsize=16)
plt.title('Original Series')
plt.show()

# Step 1: Normalize the time series
mu = np.mean(ts).iloc[0]
sigma = np.std(ts).iloc[0]
norm_ts = (ts - mu) / sigma

# Plot the normalized time series
plt.figure(figsize=(10, 4))
plt.plot(norm_ts)
plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours Since Published', fontsize=16)
plt.yticks(np.arange(-3, 2), fontsize=14)
plt.ylabel('Norm. Views', fontsize=16)
plt.title('1. Normalized Series')
plt.show()

# Step 2: Exponentiate the normalized time series
exp_ts = np.exp(norm_ts)

# Plot the exponentiated normalized time series
plt.figure(figsize=(10, 4))
plt.plot(exp_ts)
plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours Since Published', fontsize=16)
plt.yticks(np.arange(0, 3.5, .5), fontsize=14)
plt.ylabel('Exp. Norm. Views', fontsize=16)
plt.title('2. Exponentiated Normalized Series')
plt.show()

# Perform ADF test on the exponentiated normalized time series
perform_adf_test(exp_ts)
print("3. First Difference")

# Step 3: Take the first difference of the exponentiated normalized time series
diff_ts = exp_ts.diff().dropna()

# Plot the first difference of the exponentiated normalized time series
plt.figure(figsize=(10, 4))
plt.plot(diff_ts)
plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours Since Published', fontsize=16)
plt.yticks(np.arange(-0.2, 0.3, .1), fontsize=14)
plt.ylabel('First Diff. Exp. Norm. Views', fontsize=16)
plt.title('3. First Difference of Exponentiated Normalized Series')
plt.show()

# Perform ADF test on the first difference series
perform_adf_test(diff_ts)

# Fit an ARMA model to the first difference series
plot_pacf(diff_ts)
plt.title('Partial Autocorrelation Function')
plt.show()

plot_acf(diff_ts)
plt.title('Autocorrelation Function')
plt.show()

# Create the SARIMA model
order = (4, 0, 1)  # AR order, differencing order, MA order
model = ARIMA(diff_ts, order=order)
model_fit = model.fit()

# Predict the next 3 hours using the SARIMA model
forecast_steps = 3
prediction_info = model_fit.get_forecast(steps=forecast_steps)

# Retrieve the predicted mean and confidence intervals
predictions = prediction_info.predicted_mean
conf_int = prediction_info.conf_int()

# Extract lower and upper bounds for the confidence intervals
lower_bound = conf_int.iloc[:, 0]
upper_bound = conf_int.iloc[:, 1]

# Plot the original first difference series and the predicted values
plt.figure(figsize=(10, 4))
plt.plot(diff_ts)
plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours Since Published', fontsize=16)
plt.yticks(np.arange(-0.2, 0.3, .1), fontsize=14)
plt.ylabel('First Diff. Exp. Norm. Views', fontsize=16)
plt.plot(np.arange(len(ts) + 1, len(ts) + 1 + forecast_steps), predictions, color='g')
plt.fill_between(np.arange(len(ts) + 1, len(ts) + 1 + forecast_steps), lower_bound, upper_bound, color='g', alpha=0.1)
plt.title('SARIMA Model Predictions')
plt.show()


# Define a function to undo transformations and obtain original predictions
def undo_transformations(predictions, series, mu, sigma):
    """
    Undo the transformations applied to predictions to obtain original values.

    Args:
        predictions (array-like): Predicted values after transformations.
        series (pd.Series): Original time series.
        mu (float): Mean used for normalization.
        sigma (float): Standard deviation used for normalization.

    Returns:
        array-like: Original predictions.
    """
    print("Series index:", series.index)
    first_pred = sigma * np.log(predictions.iloc[0] + np.exp((series.iloc[-1] - mu) / sigma)) + mu
    orig_predictions = [first_pred]

    for i in range(len(predictions[1:])):
        next_pred = sigma * np.log(predictions.iloc[i + 1] + np.exp((orig_predictions[-1] - mu) / sigma)) + mu
        orig_predictions.append(next_pred)

    return np.array(orig_predictions).flatten()


# Undo transformations to obtain original predictions
orig_preds = undo_transformations(predictions, ts, mu, sigma)
orig_lower_bound = undo_transformations(lower_bound, ts, mu, sigma)
orig_upper_bound = undo_transformations(upper_bound, ts, mu, sigma)

# Plot the original time series with the original predictions
plt.figure(figsize=(10, 4))
plt.plot(ts)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours Since Published', fontsize=16)

plt.yticks(np.arange(0, 50000, 10000), fontsize=14)
plt.ylabel('Views', fontsize=16)

plt.plot(np.arange(len(ts) + 1, len(ts) + 4), orig_preds, color='g')
plt.fill_between(np.arange(len(ts) + 1, len(ts) + 4), orig_lower_bound, orig_upper_bound, color='g', alpha=0.1)

plt.show()

# Create a new figure with a custom size
plt.figure(figsize=(10, 4))

# Plot the original time series
plt.plot(ts)

# Set the x-axis tick positions and labels
plt.xticks(np.arange(0, 78), fontsize=14)
plt.xlabel('Hours Since Published', fontsize=16)

# Set the y-axis tick positions and labels
plt.yticks(np.arange(40000, 46000, 1000), fontsize=14)
plt.ylabel('Views', fontsize=16)

# Plot the original predictions in green
plt.plot(np.arange(len(ts) + 1, len(ts) + 4), orig_preds, color='g')

# Fill the area between upper and lower bounds with light green color
plt.fill_between(np.arange(len(ts) + 1, len(ts) + 4), orig_lower_bound, orig_upper_bound, color='g', alpha=0.1)

# Set the x-axis limit to focus on a specific range
plt.xlim(64, 76)

# Set the y-axis limit to focus on a specific range
plt.ylim(40000, 45000)

# Display the plot
plt.show()
