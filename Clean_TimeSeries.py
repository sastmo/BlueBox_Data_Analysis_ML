import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


# Read the Data and parse dates
def parser(s):
    return datetime.strptime(s, '%Y-%m')


# Read the CSV file containing the data and parse the dates
ice_cream_heater_df = pd.read_csv(
   r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\ice_cream_vs_heater.csv"
    , parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# Infer the frequency and set it accordingly
ice_cream_heater_df = ice_cream_heater_df.asfreq(pd.infer_freq(ice_cream_heater_df.index))


# Plot the original series
def plot_series(series):
    plt.figure(figsize=(12, 6))
    plt.plot(series, color='red')
    plt.ylabel('Search Frequency for "Heater"', fontsize=16)

    for year in range(2004, 2021):
        plt.axvline(datetime(year, 1, 1), linestyle='--', color='k', alpha=0.5)


plot_series(ice_cream_heater_df.heater)
plt.show()

# Normalize the data
avg, dev = ice_cream_heater_df.heater.mean(), ice_cream_heater_df.heater.std()
heater_series = (ice_cream_heater_df.heater - avg) / dev
plot_series(heater_series)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)
plt.show()

# Take the first difference to remove trend
heater_series = heater_series.diff().dropna()
plot_series(heater_series)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)
plt.show()

# Remove increasing volatility
annual_volatility = heater_series.groupby(heater_series.index.year).std()
heater_annual_vol = heater_series.index.map(lambda d: annual_volatility.loc[d.year])
heater_series = heater_series / heater_annual_vol
plot_series(heater_series)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)
plt.show()

# Remove seasonality
month_avgs = heater_series.groupby(heater_series.index.month).mean()
heater_month_avg = heater_series.index.map(lambda d: month_avgs.loc[d.month])
heater_series = heater_series - heater_month_avg
plot_series(heater_series)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

plt.show()

# Perform Augmented Dickey-Fuller test
result = adfuller(heater_series)

print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:", result[4])
