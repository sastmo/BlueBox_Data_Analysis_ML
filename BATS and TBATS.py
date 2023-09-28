# https://towardsdatascience.com/how-to-forecast-time-series-with-multiple-seasonalities-23c77152347e
# https://github.com/marcopeix/time-series-analysis/blob/master/BATS_TBATS/BATS%20and%20TBATS.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.model_selection import temporal_train_test_split

# Load and preprocess data
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\daily_traffic.csv")
data = data.dropna()

# Exploration: Plot traffic volume over time
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(data['traffic_volume'])
ax.set_xlabel('Time')
ax.set_ylabel('Traffic Volume')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Exploration: Plot traffic volume over weekdays and weekends
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(data['traffic_volume'])
ax.set_xlabel('Time')
ax.set_ylabel('Traffic volume')
plt.xticks(np.arange(7, 400, 24),
           ['Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
            'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xlim(0, 400)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Modeling: Train-test split
y = data['traffic_volume']
fh = np.arange(1, 168)
y_train, y_test = temporal_train_test_split(y, test_size=168)

# Plot train and test split
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(y_train, ls='-', label='Train')
ax.plot(y_test, ls='--', label='Test')
ax.set_xlabel('Time')
ax.set_ylabel('Daily Traffic Volume')
ax.legend(loc='best')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Baseline forecast
y_pred_baseline = y_train[-168:].values

# BATS model
forecaster_bats = BATS(use_box_cox=True, use_trend=False, use_damped_trend=False, sp=[24, 168])
forecaster_bats.fit(y_train)
y_pred_bats = forecaster_bats.predict(fh)

# TBATS model
forecaster_tbats = TBATS(use_box_cox=True, use_trend=False, use_damped_trend=False, sp=[24, 168])
forecaster_tbats.fit(y_train)
y_pred_tbats = forecaster_tbats.predict(fh)

# Plot forecasts
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(y_train, ls='-', label='Train')
ax.plot(y_test, ls='-', label='Test')
ax.plot(y_test.index, y_pred_baseline, ls=':', label='Baseline')
ax.plot(y_pred_bats, ls='--', label='BATS')
ax.plot(y_pred_bats, ls='-.', label='TBATS')
ax.set_xlabel('time')
ax.set_ylabel('Daily traffic volume')
ax.legend(loc='best')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Plot forecasts
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(y_train, ls='-', label='Train')
ax.plot(y_test, ls='-', label='Test')
ax.plot(y_test.index, y_pred_baseline, ls=':', label='Baseline')
ax.plot(y_pred_bats, ls='--', label='BATS')
ax.plot(y_pred_tbats, ls='-.', label='TBATS')
ax.set_xlabel('Time')
ax.set_ylabel('Daily Traffic Volume')
ax.legend(loc='best')
fig.autofmt_xdate()
plt.xlim(800, 1000)
plt.ylim(0, 8000)
plt.tight_layout()
plt.show()


# MAPE calculation
def mape(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)


mape_baseline = mape(y_test, y_pred_baseline)
mape_bats = mape(y_test, y_pred_bats)
mape_tbats = mape(y_test, y_pred_tbats)

print(f'MAPE from baseline: {mape_baseline}')
print(f'MAPE from BATS: {mape_bats}')
print(f'MAPE from TBATS: {mape_tbats}')

# Bar plot of MAPE values
fig, ax = plt.subplots()
models = ['Baseline', 'BATS', 'TBATS']
mape_values = [mape_baseline, mape_bats, mape_tbats]
ax.bar(models, mape_values, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('MAPE (%)')
ax.set_ylim(0, 35)

for index, value in enumerate(mape_values):
    plt.text(index, value + 1, str(value), ha='center')

plt.tight_layout()
plt.show()
