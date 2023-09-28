# https://github.com/ritvikmath/Time-Series-Analysis/blob/master/MA%20Model.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import pearsonr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta

# Register converters to handle date formats
register_matplotlib_converters()

# Generate Some Data
errors = np.random.normal(0, 1, 400)
date_index = pd.date_range(start='9/1/2019', end='1/1/2020')
mu = 50
series = []
for t in range(1, len(date_index) + 1):
    series.append(mu + 0.4 * errors[t - 1] + 0.3 * errors[t - 2] + errors[t])
series = pd.Series(series, date_index)
series = series.asfreq(pd.infer_freq(series.index))

# Visualize the generated data
plt.figure(figsize=(10, 4))
plt.plot(series)
plt.axhline(mu, linestyle='--', color='grey')
plt.title('Generated Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


# Function to calculate autocorrelation
def calc_corr(series, lag):
    return pearsonr(series[:-lag], series[lag:])[0]


# ACF
acf_vals = acf(series)
num_lags_acf = len(acf_vals)  # Use the actual length of acf_vals
plt.bar(range(num_lags_acf), acf_vals)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF Value')
plt.show()

# PACF
pacf_vals = pacf(series)
num_lags_pacf = len(pacf_vals)  # Use the actual length of pacf_vals
plt.bar(range(num_lags_pacf), pacf_vals)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('PACF Value')
plt.show()

# Get training and testing sets
train_end = datetime(2019, 12, 30)
test_end = datetime(2020, 1, 1)

train_data = series[:train_end]
test_data = series[train_end + timedelta(days=1):test_end]

# Fit ARIMA Model
# Create the model
model = SARIMAX(train_data, order=(4, 0, 0))

# Fit the model
model_fit = model.fit()

# Print summary of the model
print(model_fit.summary())

# Get prediction start and end dates
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

# Get the predictions and residuals
predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

# Visualize actual vs. predicted
plt.figure(figsize=(10, 4))
plt.plot(series[-14:], label='Data')
plt.plot(predictions, label='Predictions')
plt.legend(fontsize=16)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Date')
plt.ylabel('Value')

plt.show()

# Calculate evaluation metrics
mean_abs_percent_error = round(np.mean(abs(residuals / test_data)), 4)
root_mean_squared_error = np.sqrt(np.mean(residuals ** 2))

print('Mean Absolute Percent Error:', mean_abs_percent_error)
print('Root Mean Squared Error:', root_mean_squared_error)

# Return the evaluation metrics for further analysis
mean_abs_percent_error, root_mean_squared_error
