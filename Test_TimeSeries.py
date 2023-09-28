# https://chat.openai.com/share/8bd37eef-41a2-4348-a0c2-c9ff7d70ec00
# https://github.com/ritvikmath/Time-Series-Analysis/blob/master/Model%20Selection.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA  # Updated import
from statsmodels.tsa.stattools import adfuller

register_matplotlib_converters()


# Function to parse date
def parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d')


# Function to perform Augmented Dickey-Fuller Test
def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])


# Read data
data_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\catfish.csv"
series = pd.read_csv(data_path, parse_dates=[0], index_col=0, squeeze=True, date_parser=parse_date)

# Resample and preprocess data
series = series.asfreq(pd.infer_freq(series.index))
series = series.loc[datetime(2004, 1, 1):]
series = series.diff().diff().dropna()

# Check stationarity using Augmented Dickey-Fuller Test
perform_adf_test(series)

# Plot original time series
plt.plot(series)
plt.show()

# Plot Partial Autocorrelation Function (PACF)
plot_pacf(series, lags=10)
plt.show()

# Fit AR(p) models and compare using AIC and BIC
ar_orders = [1, 4, 6, 10]
fitted_model_dict = {}

plt.figure(figsize=(12, 12))

for idx, ar_order in enumerate(ar_orders):
    # Create AR(p) model
    ar_model = ARIMA(series, order=(ar_order, 0, 0))
    ar_model_fit = ar_model.fit()
    fitted_model_dict[ar_order] = ar_model_fit

    # Plot original and fitted values
    plt.subplot(4, 1, idx + 1)
    plt.plot(series)
    plt.plot(ar_model_fit.fittedvalues)
    plt.title(f'AR({ar_order}) Fit', fontsize=16)

plt.tight_layout()
plt.show()

# Compare AIC values for different AR(p) models
for ar_order in ar_orders:
    print(f'AIC for AR({ar_order}): {fitted_model_dict[ar_order].aic}')

# Compare BIC values for different AR(p) models
for ar_order in ar_orders:
    print(f'BIC for AR({ar_order}): {fitted_model_dict[ar_order].bic}')
