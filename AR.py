from datetime import datetime, timedelta
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AR, AutoReg

# Set up matplotlib
register_matplotlib_converters()


# Load and preprocess Ice Cream Production Data
df_ice_cream = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\ice_cream.csv")
df_ice_cream.rename(columns={'DATE': 'date', 'IPN31152N': 'production'}, inplace=True)
df_ice_cream['date'] = pd.to_datetime(df_ice_cream.date)
df_ice_cream.set_index('date', inplace=True)
df_ice_cream = df_ice_cream['2010-01-01':]

# Visualize Ice Cream Production Data
plt.figure(figsize=(10,4))
plt.plot(df_ice_cream.production)
plt.title('Ice Cream Production over Time', fontsize=20)
plt.ylabel('Production', fontsize=16)
for year in range(2011, 2021):
    plt.axvline(pd.to_datetime(f'{year}-01-01'), color='k', linestyle='--', alpha=0.2)
plt.show()

# ACF and PACF analysis for Ice Cream Production Data
plt.figure(figsize=(12, 6))
plt.subplot(121)
acf_plot = plot_acf(df_ice_cream.production, lags=100, ax=plt.gca(), title='ACF')
plt.subplot(122)
pacf_plot = plot_pacf(df_ice_cream.production, ax=plt.gca(), title='PACF')
plt.tight_layout()
plt.show()

'''
# Load and preprocess Stock Data
tickerSymbol = 'SPY'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d', start='2015-01-01', end='2020-01-01')
tickerDf = tickerDf[['Close']]

# Visualize Stock Data
plt.figure(figsize=(10,4))
plt.plot(tickerDf.Close)
plt.title(f'Stock Price over Time ({tickerSymbol})', fontsize=20)
plt.ylabel('Price', fontsize=16)
for year in range(2015, 2021):
    plt.axvline(pd.to_datetime(f'{year}-01-01'), color='k', linestyle='--', alpha=0.2)
plt.show()

# Stationarity analysis for Stock Data
first_diffs = tickerDf.Close.diff().fillna(0)
tickerDf['FirstDifference'] = first_diffs

plt.figure(figsize=(10,4))
plt.plot(tickerDf.FirstDifference)
plt.title(f'First Difference over Time ({tickerSymbol})', fontsize=20)
plt.ylabel('Price Difference', fontsize=16)
for year in range(2015, 2021):
    plt.axvline(pd.to_datetime(f'{year}-01-01'), color='k', linestyle='--', alpha=0.2)
plt.show()

# ACF and PACF analysis for First Difference Data
plt.figure(figsize=(12, 6))
plt.subplot(121)
acf_plot = plot_acf(tickerDf.FirstDifference, ax=plt.gca(), title='ACF')
plt.subplot(122)
pacf_plot = plot_pacf(tickerDf.FirstDifference, ax=plt.gca(), title='PACF')
plt.tight_layout()
plt.show()
'''
# Define train and test end dates
train_end = datetime(2018, 12, 1)
test_end = datetime(2019, 12, 1)

# Split data into train and test sets
train_data = df_ice_cream[:train_end]
test_data = df_ice_cream[train_end + timedelta(days=1): test_end]

# Define order of the autoregressive model
order = 10

# Initialize and fit AutoReg model
start = time()
model = AutoReg(train_data['production'], lags=order)
model_fit = model.fit()
end = time()
print('Model Fitting Time:', end - start)

# Print model summary
print(model_fit.summary())

# Define prediction start and end dates
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

# Generate predictions using the fitted model
predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data['production'] - predictions  # Calculate residuals

# Visualize actual vs. predicted
plt.figure(figsize=(10, 4))
plt.plot(test_data.index, test_data['production'], label='Actual')
plt.plot(predictions.index, predictions, label='Predictions')
plt.legend(fontsize=16)
plt.ylabel('Production', fontsize=16)
plt.title('Ice Cream Productions over Time', fontsize=20)
for year in range(2019, 2021):
    plt.axvline(pd.to_datetime(f'{year}-01-01'), color='k', linestyle='--', alpha=0.2)
plt.show()

# Visualize residuals
plt.figure(figsize=(10, 4))
plt.plot(test_data.index, residuals)
plt.title('Residuals from AutoReg Model', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.axhline(0, color='r', linestyle='--', alpha=0.2)
for year in range(2019, 2021):
    plt.axvline(pd.to_datetime(f'{year}-01-01'), color='k', linestyle='--', alpha=0.2)
plt.show()

# Calculate evaluation metrics
mean_abs_percent_error = round(np.mean(abs(residuals / test_data['production'])), 4)
root_mean_squared_error = np.sqrt(np.mean(residuals ** 2))

print('Mean Absolute Percent Error:', mean_abs_percent_error)
print('Root Mean Squared Error:', root_mean_squared_error)

# Perform rolling predictions one month in advance
prediction_rolling = pd.Series()
for end_date in test_data.index:
    train_data_rolling = df_ice_cream[:end_date - timedelta(days=1)]
    model_rolling = AutoReg(train_data_rolling['production'], lags=order)
    model_fit_rolling = model_rolling.fit()
    pred_rolling = model_fit_rolling.predict(end_date, end_date)
    prediction_rolling.loc[end_date] = pred_rolling.loc[end_date]

# Calculate residuals for rolling predictions
residuals_rolling = test_data['production'] - prediction_rolling

# Visualize actual vs. predicted for rolling predictions
plt.figure(figsize=(10, 4))
plt.plot(test_data.index, test_data['production'], label='Actual')
plt.plot(prediction_rolling.index, prediction_rolling, label='Rolling Predictions')
plt.legend(fontsize=16)
plt.ylabel('Production', fontsize=16)
plt.title('Ice Cream Productions over Time (Rolling Predictions)', fontsize=20)
for year in range(2019, 2021):
    plt.axvline(pd.to_datetime(f'{year}-01-01'), color='k', linestyle='--', alpha=0.2)
plt.show()

# Visualize residuals for rolling predictions
plt.figure(figsize=(10, 4))
plt.plot(test_data.index, residuals_rolling)
plt.title('Residuals from AutoReg Model (Rolling Predictions)', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.axhline(0, color='r', linestyle='--', alpha=0.2)
for year in range(2019, 2021):
    plt.axvline(pd.to_datetime(f'{year}-01-01'), color='k', linestyle='--', alpha=0.2)
plt.show()

# Calculate evaluation metrics for rolling predictions
mean_abs_percent_error_rolling = round(np.mean(abs(residuals_rolling / test_data['production'])), 4)
root_mean_squared_error_rolling = np.sqrt(np.mean(residuals_rolling ** 2))

print('Mean Absolute Percent Error (Rolling Predictions):', mean_abs_percent_error_rolling)
print('Root Mean Squared Error (Rolling Predictions):', root_mean_squared_error_rolling)
