# https://chat.openai.com/share/9058d9e7-aca1-4b27-8e8b-d04c7017ef74

from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# from
# import Prophet
import quandl
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Additive Model: ******************************************

'''
# Load the "Airline Passengers" dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
series = read_csv(url, header=0, index_col=0, parse_dates=True, squeeze=True)

# Perform seasonal decomposition assuming an additive model
result_additive = seasonal_decompose(series, model='additive')

# Plot the observed, trend, seasonal, and residual time series for the additive model
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(series, label='Observed')
plt.ylabel('# of flights')  # Add label to y-axis
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(result_additive.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(result_additive.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(result_additive.resid, label='Residual')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Multiplicative Model: ******************************************

# Load the "Airline Passengers" dataset
# Load the "Airline Passengers" dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
series = read_csv(url, header=0, index_col=0, parse_dates=True, squeeze=True)

# Perform seasonal decomposition assuming a multiplicative model
result_multiplicative = seasonal_decompose(series, model='multiplicative')

# Plot the observed, trend, seasonal, and residual time series for the multiplicative model
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(series, label='Observed')
plt.ylabel('# of flights')  # Add label to y-axis
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(result_multiplicative.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(result_multiplicative.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(result_multiplicative.resid, label='Residual')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()'''

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
import quandl

# Set your Quandl API key
QUANDL_KEY = 'ijHGskymjuswfzDnxaJc'
quandl.ApiConfig.api_key = QUANDL_KEY

# Fetch the data
df = quandl.get(dataset='WGC/GOLD_DAILY_USD', start_date='2000-01-01', end_date='2005-12-31')
df.reset_index(drop=False, inplace=True)
df.rename(columns={'Date': 'ds', 'Value': 'y'}, inplace=True)

# Split the series into training and test sets
train_indices = df['ds'].apply(lambda x: x.year) < 2005
df_train = df.loc[train_indices].dropna()
df_test = df.loc[~train_indices].reset_index(drop=True)

# Extracting the time series data
values_train = df_train['y']

# Create the instance of the model and fit it to the data
model_stats = ExponentialSmoothing(values_train, seasonal='add', seasonal_periods=12)
fitted_model = model_stats.fit()

# Forecast the gold prices and create future dates
forecast_periods = 365
forecast_values = fitted_model.forecast(steps=forecast_periods)
future_dates = pd.date_range(start=df_train['ds'].max(), periods=forecast_periods + 1, closed='right')
forecast_df = pd.DataFrame({'ds': future_dates, 'y': forecast_values})

# Perform seasonal decomposition
seasonal_decomposition_additive = sm.tsa.seasonal_decompose(values_train, model='additive', period=12)
seasonal_decomposition_multiplicative = sm.tsa.seasonal_decompose(values_train, model='multiplicative', period=12)

# Plotting the results
plt.figure(figsize=(12, 20))

# Autocorrelation plot
plt.subplot(5, 1, 1)
plt.acorr(values_train, maxlags=30, linestyle='-', color='blue')
plt.title('Autocorrelation Plot')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')

# Time series and forecast plot
plt.subplot(5, 1, 2)
plt.plot(df_train['ds'], values_train, label='Training Data')
plt.plot(df_test['ds'], df_test['y'], label='Test Data')
plt.plot(forecast_df['ds'], forecast_df['y'], label='Forecast', linestyle='dashed')
plt.title('Time Series Forecasting using statsmodels')
plt.xlabel('Date')
plt.ylabel('Gold Prices')
plt.legend()

# Seasonal decomposition plots (Additive Model)
plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)
plt.plot(values_train, label='Observed')
plt.ylabel('Gold Prices')
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(seasonal_decomposition_additive.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(seasonal_decomposition_additive.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(seasonal_decomposition_additive.resid, label='Residual')
plt.legend(loc='upper left')

plt.tight_layout()

# Seasonal decomposition plots (Multiplicative Model)
plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)
plt.plot(values_train, label='Observed')
plt.ylabel('Gold Prices')
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(seasonal_decomposition_multiplicative.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(seasonal_decomposition_multiplicative.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(seasonal_decomposition_multiplicative.resid, label='Residual')
plt.legend(loc='upper left')

plt.tight_layout()

plt.show()

# Dickey-Fuller test
# Perform Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller


def perform_dickey_fuller_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])


print(perform_dickey_fuller_test(values_train))

# Define the window size for the moving average
window_size = 30  # For example, using a 30-days window

# Compute the moving average using the rolling function
df['moving_average'] = df['y'].rolling(window=window_size).mean()

# Plot the original time series and the moving average
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='Original Time Series')
plt.plot(df['ds'], df['moving_average'], label=f'Moving Average (Window={window_size})', color='green')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Moving Average Model')
plt.legend()
plt.show()

# Perform exponential smoothing
model_exp = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=30)
fitted_exp = model_exp.fit()
forecast_exp = fitted_exp.forecast(steps=365)

# Perform double exponential smoothing
model_double_exp = ExponentialSmoothing(df['y'], trend='add', seasonal=None)
fitted_double_exp = model_double_exp.fit()
forecast_double_exp = fitted_double_exp.forecast(steps=365)

# Perform triple exponential smoothing
model_triple_exp = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=30)
fitted_triple_exp = model_triple_exp.fit()
forecast_triple_exp = fitted_triple_exp.forecast(steps=365)

# Plotting
plt.figure(figsize=(12, 15))

# Original Gold Prices
plt.subplot(3, 1, 1)
plt.plot(df['ds'], df['y'], label='Original Gold Prices')
plt.title('Original Gold Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

# Exponential Smoothing
plt.subplot(3, 1, 2)
plt.plot(df['ds'], df['y'], label='Original Gold Prices')
plt.plot(df['ds'].iloc[-1] + pd.DateOffset(days=1) + pd.to_timedelta(range(365), unit='D'), forecast_exp, label='Forecast')
plt.title('Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

# Double Exponential Smoothing
plt.subplot(3, 1, 3)
plt.plot(df['ds'], df['y'], label='Original Gold Prices')
plt.plot(df['ds'].iloc[-1] + pd.DateOffset(days=1) + pd.to_timedelta(range(365), unit='D'), forecast_double_exp, label='Forecast')
plt.title('Double Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

plt.tight_layout()
plt.show()
