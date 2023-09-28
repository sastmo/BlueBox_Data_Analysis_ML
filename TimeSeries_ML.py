# https://towardsdatascience.com/the-complete-guide-to-time-series-forecasting-using-sklearn-pandas-and-numpy-7694c90e45c1
# https://towardsdatascience.com/is-gradient-boosting-good-as-prophet-for-time-series-forecasting-3dcbfd03775e
# https://medium.com/towards-data-science/time-series-nested-cross-validation-76adba623eb9
# https://chat.openai.com/share/000eec5e-438f-45fc-84a5-1d9f74c12fa9

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load CO2 concentration data
data = sm.datasets.co2.load_pandas().data

# Create a copy of the DataFrame
df = data.copy()

# Calculate first differences to make the time series stationary
data_diff = data['co2'].diff().dropna()

# Create a figure and axis for the CO2 concentration plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(data.index, data['co2'], label='CO2 concentration')
ax.set_xlabel('Time')
ax.set_ylabel('CO2 Concentration (ppmw)')
ax.set_title('CO2 Concentration Over Time')
ax.legend()
fig.autofmt_xdate()  # Format x-axis date labels
plt.tight_layout()  # Improve layout and spacing
plt.show()  # Display the CO2 concentration plot


# Create ACF plot for the differenced data
plt.figure(figsize=(10, 4))
plot_acf(data_diff)
plt.title('ACF Plot', fontsize=20)
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.xticks(np.arange(0, 25, 1))
plt.show()

# Create PACF plot for the differenced data
plt.figure(figsize=(10, 4))
plot_pacf(data_diff)
plt.title('PACF Plot', fontsize=20)
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.xticks(np.arange(0, 25, 1))
plt.show()


# Create a new column 'x1' with CO2 values shifted by one position
df['x1'] = df['co2'].shift(-1)
df['x2'] = df['x1'].shift(-1)
df['y'] = df['x2'].shift(-1)

# Drop rows with null values
df_cleaned = df.dropna()

# Split the data into training and test sets
train = df_cleaned.iloc[:-104]
test = df_cleaned.iloc[-104:]

# Create a copy of the test DataFrame
test_copy = test.copy()

# Create a new column 'baseline_pred' and set it as the 'co2' values
test_copy['baseline_pred'] = test_copy['co2']

# Drop the last rows from the test_copy DataFrame
test_copy = test_copy.drop(test_copy.tail(3).index)

# Reshape training and test data
X_train = train[['co2', 'x1', 'x2']].values
y_train = train['y'].values.reshape(-1, 1)
X_test = test_copy[['co2', 'x1', 'x2']].values
y_test = test_copy['y'].values.reshape(-1, 1)

# Initialize the DecisionTreeRegressor model
dt_reg = DecisionTreeRegressor(random_state=42)

# Fit the model on the training data
dt_reg.fit(X=X_train, y=y_train)

# Make predictions using the trained model
dt_pred = dt_reg.predict(X_test)

# Assign the predictions to a new column 'dt_pred' in the test_copy DataFrame
test_copy['dt_pred'] = dt_pred

# Display the modified test_copy DataFrame
print(test_copy)

# Initialize GradientBoostingRegressor model
gbr = GradientBoostingRegressor(random_state=42)

# Fit the model on the training data
gbr.fit(X_train, y=y_train.ravel())

# Make predictions using the trained model
gbr_pred = gbr.predict(X_test)

# Assign the predictions to a new column 'gbr_pred' in the test_copy DataFrame
test_copy['gbr_pred'] = gbr_pred

# Display the modified test_copy DataFrame
print(test_copy)

# Calculate evaluation metrics
# Random Forest
mse_r = mean_squared_error(y_test, dt_pred)
rmse_r = mean_squared_error(y_test, dt_pred, squared=False)
mae_r = mean_absolute_error(y_test, dt_pred)
r2_r = r2_score(y_test, dt_pred)

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse_r}")
print(f"Root Mean Squared Error (RMSE): {rmse_r}")
print(f"Mean Absolute Error (MAE): {mae_r}")
print(f"R-squared (R2): {r2_r}")
print("-_-" * 30)

# Baseline
mse = mean_squared_error(y_test, test_copy['baseline_pred'])
rmse = mean_squared_error(y_test, test_copy['baseline_pred'], squared=False)
mae = mean_absolute_error(y_test, test_copy['baseline_pred'])
r2 = r2_score(y_test, test_copy['baseline_pred'])

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")
print("-_-" * 30)

# GradientBoost
mse_g = mean_squared_error(y_test, test_copy['gbr_pred'])
rmse_g = mean_squared_error(y_test, test_copy['gbr_pred'], squared=False)
mae_g = mean_absolute_error(y_test, test_copy['gbr_pred'])
r2_g = r2_score(y_test, test_copy['gbr_pred'])

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse_g}")
print(f"Root Mean Squared Error (RMSE): {rmse_g}")
print(f"Mean Absolute Error (MAE): {mae_g}")
print(f"R-squared (R2): {r2_g}")
print("-_-" * 30)

error_table = [mae, mae_r, mae_g]
labels = ['MAE', 'MAE_R', 'MAE_G']

fig, ax = plt.subplots(figsize=(7, 5))
bars = plt.bar(labels, error_table)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Some X Label')
plt.ylabel('Some Y Label')
plt.title('Error Comparison')
plt.legend(labels)


# plt.show()


def window_input(window_length: int, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    i = 1
    while i < window_length:
        df[f'x_{i}'] = df['co2'].shift(-i)
        i = i + 1

    if i == window_length:
        df['y'] = df['co2'].shift(-i)

    # Drop rows where there is a NaN
    df = df.dropna(axis=0)

    return df


new_df = window_input(5, data)
print(new_df)

# Baseline model
X = new_df[['co2', 'x_1', 'x_2', 'x_3', 'x_4']].values
y = new_df['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

# Concatenate X_test and y_test to create pred_df
pred_df = pd.concat([pd.DataFrame(X_test), pd.Series(y_test, name='y')], axis=1)

# Apply a decision tree
dt_reg_5 = DecisionTreeRegressor(random_state=42)

dt_reg_5.fit(X_train, y_train)

dt_reg_5_pred = dt_reg_5.predict(X_test)

pred_df['dt_pred5'] = dt_reg_5_pred

# Apply gradient boosting
gbr_5 = GradientBoostingRegressor(random_state=42)

gbr_5.fit(X_train, y_train.ravel())

gbr_5_pred = gbr_5.predict(X_test)

pred_df['gbr_pred5'] = gbr_5_pred

print(pred_df)

# Calculate evaluation metrics
# Random Forest
mse_r5 = mean_squared_error(y_test, dt_reg_5_pred)
rmse_r5 = mean_squared_error(y_test, dt_reg_5_pred, squared=False)
mae_r5 = mean_absolute_error(y_test, dt_reg_5_pred)
r2_r5 = r2_score(y_test, dt_reg_5_pred)

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse_r5}")
print(f"Root Mean Squared Error (RMSE): {rmse_r5}")
print(f"Mean Absolute Error (MAE): {mae_r5}")
print(f"R-squared (R2): {r2_r5}")
print("-_-" * 30)

# GradientBoost
mse_g5 = mean_squared_error(y_test, gbr_5_pred)
rmse_g5 = mean_squared_error(y_test, gbr_5_pred, squared=False)
mae_g5 = mean_absolute_error(y_test, gbr_5_pred)
r2_g5 = r2_score(y_test, gbr_5_pred)

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse_g5}")
print(f"Root Mean Squared Error (RMSE): {rmse_g5}")
print(f"Mean Absolute Error (MAE): {mae_g5}")
print(f"R-squared (R2): {r2_g5}")
print("-_-" * 30)

error_table = [mae, mae_r5, mae_g5]
labels = ['MAE', 'MAE_R5', 'MAE_G5']

fig, ax = plt.subplots(figsize=(7, 5))
bars = plt.bar(labels, error_table)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Some X Label')
plt.ylabel('Some Y Label')
plt.title('Error Comparison')
plt.legend(labels)
plt.show()

# Create a figure and axis for the CO2 concentration plot
# Create a figure and axis for the CO2 concentration plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot CO2 concentration from pred_df, dt_pred5, and gbr_pred5
ax.plot(pred_df.index, pred_df['dt_pred5'], label='Decision Tree Prediction')
ax.plot(pred_df.index, pred_df['gbr_pred5'], label='Gradient Boosting Prediction')

# Plot original CO2 concentration from data
ax.plot(pd.DataFrame(y_test).index, pd.DataFrame(y_test), label='Original CO2 concentration', linestyle='dashed')

ax.set_xlabel('Time')
ax.set_ylabel('CO2 Concentration (ppmw)')
ax.set_title('CO2 Concentration Over Time')
ax.legend()
fig.autofmt_xdate()  # Format x-axis date labels
plt.tight_layout()  # Improve layout and spacing
plt.show()  # Display the CO2 concentration plot

# Predict a sequence


def window_input_output(input_length: int, output_length: int, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    i = 1
    while i < input_length:
        df[f'x_{i}'] = df['co2'].shift(-i)
        i = i + 1

    j = 0
    while j < output_length:
        df[f'y_{j}'] = df['co2'].shift(-output_length - j)
        j = j + 1

    df = df.dropna(axis=0)

    return df


# Assuming data is defined
seq_df = window_input_output(26, 26, data)

# Selecting columns for predictors (X) and targets (y)
X_cols = [col for col in seq_df.columns if col.startswith('x')]
X_cols.insert(0, 'co2')  # Adding 'co2' as a predictor
y_cols = [col for col in seq_df.columns if col.startswith('y')]

# Splitting the data into training and test sets
X_train = seq_df[X_cols][:-2].values
y_train = seq_df[y_cols][:-2].values
X_test = seq_df[X_cols][-2:].values
y_test = seq_df[y_cols][-2:].values

# Decision Tree model
dt_seq = DecisionTreeRegressor(random_state=42)
dt_seq.fit(X_train, y_train)
dt_seq_preds = dt_seq.predict(X_test)

# Gradient Boosting model with RegressorChain wrapper
from sklearn.multioutput import RegressorChain

gbr_seq = GradientBoostingRegressor(random_state=42)
chained_gbr = RegressorChain(gbr_seq)
chained_gbr.fit(X_train, y_train)
gbr_seq_preds = chained_gbr.predict(X_test)

# Visualization of predictions over the last year
fig, ax = plt.subplots(figsize=(16, 11))
ax.plot(np.arange(0, 26, 1), X_test[1], 'b-', label='input')
ax.plot(np.arange(26, 52, 1), y_test[1], marker='.', color='blue', label='Actual')
ax.plot(np.arange(26, 52, 1), X_test[1], marker='o', color='red', label='Baseline')
ax.plot(np.arange(26, 52, 1), dt_seq_preds[1], marker='^', color='green', label='Decision Tree')
ax.plot(np.arange(26, 52, 1), gbr_seq_preds[1], marker='P', color='black', label='Gradient Boosting')
ax.set_xlabel('Timesteps')
ax.set_ylabel('CO2 concentration (ppmv)')
plt.xticks(np.arange(1, 104, 52), np.arange(2000, 2002, 1))
plt.legend(loc=2)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
