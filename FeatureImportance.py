# https://medium.com/@ali.soleymani.co/stop-using-random-forest-feature-importances-take-this-intuitive-approach-instead-4335205b933f
# https://chat.openai.com/share/a2f33504-f51f-4bf9-b522-06ef5d8bb95f

from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# Read the CSV file
file_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\Etsy.csv"
missing_values = ["na", "nan", "na.na", "n.a", 0]
shopping = pd.read_csv(file_path, na_values=missing_values)

# Impute missing values
imputed = SimpleImputer(strategy="most_frequent")
data_imputed = pd.DataFrame(imputed.fit_transform(shopping), columns=shopping.columns)

# Convert 'open_date' column to datetime format
data_imputed['open_date'] = pd.to_datetime(data_imputed['open_date'])

# Calculate day differences from the maximum timestamp
data_imputed['age'] = (data_imputed['open_date'].max() - data_imputed['open_date']).dt.days

# Convert age column to integer type
data_imputed['age'] = data_imputed['age'].astype(int)

# Plotting histograms
columns_to_plot = ['feedback', 'admirers', 'age', 'items']
num_rows = 2
num_cols = 2
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))
for i, column in enumerate(columns_to_plot):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.hist(data_imputed[column], color='blue', alpha=0.5, bins=300)
    ax.set_xlabel(column.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.set_title(f'{column.replace("_", " ").title()} Histogram')
plt.tight_layout()
plt.show()

# Plotting sales histogram
plt.figure(figsize=(10, 6))
plt.hist(shopping['sales'], color='green', alpha=0.5, bins=300)  # Changed from scatter to hist
plt.xlabel('Number of sales')
plt.ylabel('Frequency')
plt.title('Sales Histogram')
plt.show()

# Features Correlation with Sales
columns_to_plot = ['feedback', 'admirers', 'age', 'items']
num_rows = 2
num_cols = 2

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))

# Loop through columns and plot scatter plots in the grid
for i, column in enumerate(columns_to_plot):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]

    # Scatter plot of the current column against sales
    ax.scatter(data_imputed[column], data_imputed['sales'], color='blue', alpha=0.5)

    # Set labels and title
    ax.set_xlabel(column.replace('_', ' ').title())  # Formatted xlabel
    ax.set_ylabel('Sales')
    ax.set_title(f'Sales by {column.replace("_", " ").title()}')  # Formatted title

plt.tight_layout()
plt.show()

# List of columns to calculate correlation with
columns_to_calculate_corr = ['sales', 'feedback', 'admirers', 'age', 'items']

# Filter out non-numeric columns
numeric_columns = [col for col in columns_to_calculate_corr if pd.api.types.is_numeric_dtype(data_imputed[col])]

# Calculate the correlation matrix
correlation_matrix = data_imputed[numeric_columns].corr()

# Display the correlation matrix
print(correlation_matrix)

# Build the Random Forest model ****************************************************

# Prepare the data
X = data_imputed[['feedback', 'admirers', 'age', 'items']]
y = data_imputed['sales']

# Split the data into training and testing sets
X_train_, X_test_, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the scaler on the training features
X_train = scaler.fit_transform(X_train_)

# Transform the test features using the same scaler
X_test = scaler.transform(X_test_)

# Create and fit the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")
print("-_-" * 27)

# Plot actual sales vs. predicted values with different colors
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.7, label='Predicted')
plt.scatter(y_test, y_test, color='red', alpha=0.7, label='Actual')  # Actual values
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.show()

# Improving Results with K Cross Validation & Hyperparameter Tuning
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, None],
    'max_features': ["sqrt", "log2", None],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [10, 25, 50, 75, 100]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Checking for Best Hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Create a new RandomForestRegressor instance with the best hyperparameters
best_grid = grid_search.best_estimator_
y_pred_train_best = best_grid.predict(X_train)
y_pred_best = best_grid.predict(X_test)

# Assume a New Data Set, and we should predict the salse *******************************

# New Data
new_data = pd.DataFrame({
    'feedback': [4.5, 3.8, 4.2],
    'admirers': [1500, 1200, 1800],
    'date_': ['2012-08-15', '2013-09-10', '2012-10-23'],
    'open_year': [2012, 2012, 2013],
    'open_month': [8, 9, 10],
    'open_day': [15, 10, 5],
    'items': [200, 250, 180]
})

new_data['open_date'] = pd.to_datetime(new_data['date_'])

# Calculate day differences from the maximum timestamp
new_data['age'] = (new_data['open_date'].max() - new_data['open_date']).dt.days

# Use the trained model to make predictions on the new data
new_X_ = new_data[['feedback', 'admirers', 'age', 'items']]
new_X = scaler.transform(new_X_)
new_predictions = best_grid.predict(new_X)

# Display the predicted sales for the new data
for i, prediction in enumerate(new_predictions):
    print(f"Prediction for sample {i + 1}: {prediction}")
print("-_-" * 27)

# Get feature importance scores **************************************************
importance_scores = best_grid.feature_importances_

# Get the corresponding feature names
feature_names = X.columns  # Assuming X contains your predictor variables

# Create a sorted index based on importance scores
sorted_index = importance_scores.argsort()

# Create a DataFrame with feature names and importance scores
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})

# Sort the DataFrame by importance scores
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance DataFrame
print("Feature Importance using feature_importances_ method", feature_importance_df)
print("-_-" * 27)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_index)), importance_scores[sorted_index], align="center")
plt.yticks(range(len(sorted_index)), [feature_names[i] for i in sorted_index])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Scores")
plt.show()

# Permutation Feature Importance **********************************************

# Calculate permutation importance's
FeatureImp_Perm = permutation_importance(best_grid, X_test, y_test, n_repeats=10, random_state=0)

# Calculate average importance (mean of the normalized importance values)
avg_importance = FeatureImp_Perm.importances_mean / FeatureImp_Perm.importances_mean.sum()

'''
# Create a DataFrame to store the results
perm_df = pd.DataFrame({'Feature': X_train.columns,
                        'AVG_Importance': FeatureImp_Perm.importances_mean,
                        'STD_Importance': FeatureImp_Perm.importances_std})
'''

# Create a DataFrame to store the results
perm_df = pd.DataFrame({'Feature': X_train_.columns,
                        'AVG_Importance': avg_importance})


# Sort the DataFrame by average importance
perm_df = perm_df.sort_values(by='AVG_Importance', ascending=False)

# Display the feature importance DataFrame
print("Feature Importance using Permutation", perm_df)

# Plot the results
plt.figure(figsize=(10, 6))
plt.barh(perm_df['Feature'], perm_df['AVG_Importance'])
plt.xlabel('Average Importance')
plt.ylabel('Feature')
plt.title('Permutation Feature Importance')
plt.show()
