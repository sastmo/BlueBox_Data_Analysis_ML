# https://github.com/ritvikmath/Time-Series-Analysis/blob/master/ARMA%20Model.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA  # Updated import

register_matplotlib_converters()
from time import time


# Catfish Sales Data
def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')


# Read data
catfish_sales = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\catfish.csv",
                            parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# Infer the frequency of the data
catfish_sales = catfish_sales.asfreq(pd.infer_freq(catfish_sales.index))
start_date = datetime(2000, 1, 1)
end_date = datetime(2004, 1, 1)
lim_catfish_sales = catfish_sales[start_date:end_date]

# Plot the data
plt.figure(figsize=(10, 4))
plt.plot(lim_catfish_sales)
plt.title('Catfish Sales in 1000s of Pounds', fontsize=20)
plt.ylabel('Sales', fontsize=16)
for year in range(start_date.year, end_date.year):
    plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
plt.axhline(lim_catfish_sales.mean(), color='r', alpha=0.2, linestyle='--')

plt.show()

# Calculate the first difference
first_diff = lim_catfish_sales.diff()[1:]

# Plot the first difference
plt.figure(figsize=(10, 4))
plt.plot(first_diff)
plt.title('First Difference of Catfish Sales', fontsize=20)
plt.ylabel('Sales', fontsize=16)
for year in range(start_date.year, end_date.year):
    plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
plt.axhline(first_diff.mean(), color='r', alpha=0.2, linestyle='--')

plt.show()

# ACF
num_lags = 20
acf_vals = acf(first_diff, nlags=num_lags)
plt.bar(range(num_lags), acf_vals[:num_lags])
plt.title('ACF of First Difference', fontsize=20)
plt.xlabel('Lags', fontsize=16)
plt.ylabel('ACF', fontsize=16)
plt.show()

# PACF
pacf_vals = pacf(first_diff, nlags=num_lags)
plt.figure(figsize=(10, 4))
plt.bar(range(num_lags), pacf_vals[:num_lags])
plt.title('PACF of First Difference', fontsize=20)
plt.xlabel('Lags', fontsize=16)
plt.ylabel('PACF', fontsize=16)
plt.show()

# Get training and testing sets
train_end = datetime(2003, 7, 1)
test_end = datetime(2004, 1, 1)

train_data = first_diff[:train_end]
test_data = first_diff[train_end + timedelta(days=1):test_end]

# Fit the ARIMA Model
order = (4, 0, 1)  # AR order, differencing order, MA order
model = ARIMA(train_data, order=order)
start = time()
model_fit = model.fit()
end = time()
print('Model Fitting Time:', end - start)

# Summary of the model
print(model_fit.summary())

# Get prediction start and end dates
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

# Get the predictions and residuals
predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

# Plot residuals
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title('Residuals from ARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.axhline(0, color='r', linestyle='--', alpha=0.2)
plt.show()

# Plot actual vs. predicted
plt.figure(figsize=(10, 4))
plt.plot(test_data)
plt.plot(predictions)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title('First Difference of Catfish Sales (Predictions)', fontsize=20)
plt.ylabel('Sales', fontsize=16)
plt.show()

# Calculate and print Root Mean Squared Error
rmse = np.sqrt(np.mean(residuals ** 2))
print('Root Mean Squared Error:', rmse)

