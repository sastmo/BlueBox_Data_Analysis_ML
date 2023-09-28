import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Read the data
def parser(s):
    return datetime.strptime(s, '%Y-%m')


# Read the CSV file containing the data and parse the dates
ice_cream_heater_df = pd.read_csv(
    r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\ice_cream_vs_heater.csv"
    , parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# Infer the frequency and set the data frequency accordingly
ice_cream_heater_df = ice_cream_heater_df.asfreq(pd.infer_freq(ice_cream_heater_df.index))
print(ice_cream_heater_df[ice_cream_heater_df['heater'].isna() | ice_cream_heater_df['ice cream'].isna()])

# Plot the raw data
plt.figure(figsize=(12, 6))
ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
heater, = plt.plot(ice_cream_heater_df['heater'], color='red')
for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle='--', color='k', alpha=0.5)
plt.title('Plot the raw data', fontsize=20)
plt.legend(['Ice Cream', 'Heater'], fontsize=16)
plt.show()

# Normalize the data
'''
def normalize_fuc(data_set):
    avgs = data_set.mean()
    devs = data_set.std()
    for col in data_set.columns:
        data_set[col] = (data_set[col] - avgs.loc[col]) / devs.loc[col]
    return pd.DataFrame(data_set)

ice_cream_heater_df = normalize_fuc(ice_cream_heater_df)
print(ice_cream_heater_df.head())
'''
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit and transform the scaler on the data
normalized_data = scaler.fit_transform(ice_cream_heater_df)

# Convert the normalized data back to a DataFrame
ice_cream_heater_df = pd.DataFrame(normalized_data, columns=ice_cream_heater_df.columns,
                                   index=ice_cream_heater_df.index)
print(ice_cream_heater_df.head())

# Plot the normalized data
plt.figure(figsize=(12, 6))
ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
heater, = plt.plot(ice_cream_heater_df['heater'], color='red')
for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle='--', color='k', alpha=0.5)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)
plt.title('Plot the normalized data', fontsize=20)
plt.legend(['Ice Cream', 'Heater'], fontsize=16)
plt.show()

# Take the first difference to remove trend
ice_cream_heater_df = ice_cream_heater_df.diff().dropna()  # NaN values was added in Infer the frequency should drop
plt.figure(figsize=(12, 6))
ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
heater, = plt.plot(ice_cream_heater_df['heater'], color='red')
for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle='--', color='k', alpha=0.5)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)
plt.ylabel('First Difference', fontsize=18)
plt.title('Plot the first difference', fontsize=20)
plt.legend(['Ice Cream', 'Heater'], fontsize=16)
plt.show()

# Remove increasing volatility

# Remove increasing volatility for 'heater'
annual_volatility_heater = ice_cream_heater_df['heater'].groupby(ice_cream_heater_df.index.year).std()
heater_annual_vol = ice_cream_heater_df['heater'].index.map(lambda d: annual_volatility_heater.loc[d.year])
heater_series = ice_cream_heater_df['heater'] / heater_annual_vol

# Remove increasing volatility for 'ice cream'
annual_volatility_ice_cream = ice_cream_heater_df['ice cream'].groupby(ice_cream_heater_df.index.year).std()
ice_cream_annual_vol = ice_cream_heater_df['ice cream'].index.map(lambda d: annual_volatility_ice_cream.loc[d.year])
ice_cream_series = ice_cream_heater_df['ice cream'] / ice_cream_annual_vol

# Plot the modified series
plt.figure(figsize=(12, 6))
plt.plot(heater_series, color='red', label='Heater')
plt.plot(ice_cream_series, color='blue', label='Ice Cream')
plt.ylabel('Normalized Search Frequency', fontsize=16)

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle='--', color='k', alpha=0.5)

plt.axhline(0, linestyle='--', color='k', alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# Remove seasonality by subtracting monthly averages

# Plot ACF of the time series
plt.figure(figsize=(10, 4))
plot_acf(ice_cream_heater_df['heater'])
plt.title('ACF Plot of ice_cream_heater_df[heater]', fontsize=20)
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.xticks([i + 1 for i in range(24)])
plt.show()

# Plot PACF of the time series
plt.figure(figsize=(10, 4))
plot_pacf(ice_cream_heater_df['heater'])
plt.title('PACF Plot of ice_cream_heater_df[heater]', fontsize=20)
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.xticks([i + 1 for i in range(24)])
plt.show()

# Plot ACF of the time series
plt.figure(figsize=(10, 4))
plot_acf(ice_cream_heater_df['ice cream'])
plt.title('ACF Plot of ice_cream_heater_df[ice cream]', fontsize=20)
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.xticks([i + 1 for i in range(24)])
plt.show()

# Plot PACF of the time series
plt.figure(figsize=(10, 4))
plot_pacf(ice_cream_heater_df['ice cream'])
plt.title('PACF Plot of ice_cream_heater_df[ice cream]', fontsize=20)
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.xticks([i + 1 for i in range(24)])
plt.show()

# Calculate the lagged values for the same month in the previous year
'''
# Calculate the minimum year in the dataset
min_year = ice_cream_heater_df.index.min().year

# Define a function to handle the lag calculation
def lag_function(group):
    if group.index.year.min() != min_year:
        return group.shift(1)
    else:
        return group - group  # Subtract the group from itself (results in zeros)


# Calculate lagged values for each month, considering the minimum year
lagged_values = ice_cream_heater_df.groupby(ice_cream_heater_df.index.month).apply(lag_function)

# Subtract the lagged values from the original values
ice_cream_heater_df_ = ice_cream_heater_df - lagged_values

# Print the resulting DataFrame
print(ice_cream_heater_df_)
'''

# Calculate the average values for each month
month_avg = ice_cream_heater_df.groupby(ice_cream_heater_df.index.month).mean()

# Subtract the monthly averages from the 'ice cream' column
ice_cream_heater_df['ice cream'] = ice_cream_heater_df['ice cream'] - ice_cream_heater_df.index.map(
    lambda d: month_avg.loc[d.month, 'ice cream'])

# Subtract the monthly averages from the 'heater' column
ice_cream_heater_df['heater'] = ice_cream_heater_df['heater'] - ice_cream_heater_df.index.map(
    lambda d: month_avg.loc[d.month, 'heater'])

# Plot the data after removing seasonality
plt.figure(figsize=(12, 6))
ice_cream, = plt.plot(ice_cream_heater_df['ice cream'])
heater, = plt.plot(ice_cream_heater_df['heater'], color='red')
for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle='--', color='k', alpha=0.5)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)
plt.ylabel('First Difference', fontsize=18)
plt.legend(['Ice Cream', 'Heater'], fontsize=16)
plt.title(' Plot of ice_cream_heater_df', fontsize=20)
plt.show()

# PACF - Heater
# Plot PACF of the time series
plt.figure(figsize=(10, 4))
plot_pacf(ice_cream_heater_df['heater'])
plt.title('PACF Plot of ice_cream_heater_df[heater]', fontsize=20)
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.xticks([i + 1 for i in range(24)])
plt.show()

# Calculate correlation between heater and lagged ice cream
for lag in range(1, 14):
    heater_series = ice_cream_heater_df['heater'].iloc[lag:]
    lagged_ice_cream_series = ice_cream_heater_df['ice cream'].iloc[:-lag]
    print('Lag: %s' % lag)
    print(pearsonr(heater_series, lagged_ice_cream_series))
    print('------')

# Fit a VAR model
ice_cream_heater_df = ice_cream_heater_df[['ice cream', 'heater']]
model = VAR(ice_cream_heater_df)
model_fit = model.fit(maxlags=13)
model_fit.summary()

# Display the summary of the fitted VAR model
print(model_fit.summary())

# Make predictions using the fitted VAR model
forecast_steps = 3  # Change the number of forecasted steps as needed
predictions = model_fit.forecast(ice_cream_heater_df.values[-model_fit.k_ar:], steps=forecast_steps)

print(predictions, type(predictions), predictions.shape)


# Define a function to undo transformations and obtain original predictions
# Define a function to undo transformations and obtain original predictions
def undo_transformations(predictions, scaler, first_diff_values, volatility_values):
    """
    Undo the transformations applied to predictions to obtain original values.

    Args:
        predictions (array-like): Predicted values after transformations.
        scaler (StandardScaler): Scaler used for normalization.
        first_diff_values (pd.Series): First difference values of the original time series.
        volatility_values (pd.Series): Adjusted values for removing increasing volatility.
        seasonality_value (float): Single value for seasonality adjustment.

    Returns:
        array-like: Original predictions.
    """
    orig_predictions = []
    last_original_value = first_diff_values.iloc[-1]  # Use last step of original data for first prediction

    for i in range(len(predictions)):
        # Combine transformations into a single formula
        pred = predictions[i] * scaler.scale_[0] + scaler.mean_[0]
        pred -= orig_predictions[-1] if i > 0 else last_original_value
        pred /= volatility_values.iloc[i]

        orig_predictions.append(pred)

    return np.array(orig_predictions)


# Undo transformations to obtain original predictions
orig_preds = undo_transformations(predictions, scaler,
                                  ice_cream_heater_df['ice cream'].diff().dropna(),
                                  heater_series)

# Plot the original time series with the original predictions
plt.figure(figsize=(12, 6))
plt.plot(ice_cream_heater_df.index[-forecast_steps:], ice_cream_heater_df['ice cream'].values[-forecast_steps:])
plt.plot(ice_cream_heater_df.index[-forecast_steps:], orig_preds, color='g')
plt.fill_between(ice_cream_heater_df.index[-forecast_steps:], orig_preds.flatten(), orig_preds.flatten(), color='g', alpha=0.1)  # Use flatten()
plt.title('Original Time Series with VAR Model Predictions')
plt.legend(['Ice Cream', 'Predicted Ice Cream'], fontsize=12)
plt.show()