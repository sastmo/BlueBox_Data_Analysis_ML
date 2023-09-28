from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, r2_score, \
    euclidean_distances
import numpy as np
from scipy.spatial.distance import squareform
from sklearn.utils import resample

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Assume a New Data Set, and we should predict the salse

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
new_X = new_data[['feedback', 'admirers', 'age', 'items']]
new_predictions = model.predict(new_X)

# Display the predicted sales for the new data
for i, prediction in enumerate(new_predictions):
    print(f"Prediction for sample {i + 1}: {prediction}")
print("-_-" * 27)

# Get feature importance scores
importance_scores = model.feature_importances_

# Get the corresponding feature names
feature_names = X.columns  # Assuming X contains your predictor variables

# Create a sorted index based on importance scores
sorted_index = importance_scores.argsort()

# Create a DataFrame with feature names and importance scores
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})

# Sort the DataFrame by importance scores
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance DataFrame
print(feature_importance_df)
print("-_-" * 27)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_index)), importance_scores[sorted_index], align="center")
plt.yticks(range(len(sorted_index)), [feature_names[i] for i in sorted_index])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Scores")
plt.show()

# Visualize OOB Error
'''
# Create a figure object and set the size
plt.figure(figsize=(10, 6))

# Range of `n_estimators` values to explore
min_estimators = 100
max_estimators = 250

# Define a list of ensemble classifiers with their labels
ensemble_clfs = [("Random Forest (10 max_depth)", RandomForestRegressor(max_depth=20, oob_score=True, random_state=42))]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1, 5):
        clf.set_params(n_estimators=i)
        clf.fit(X_train, y_train)

        # Record the OOB error for each `n_estimators=i` setting
        oob_error = 1 - clf.oob_score
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("Number of Trees")
plt.ylabel("OOB Error Rate")
plt.title("OOB Error Rate for Random Forest")
plt.legend(loc="upper right")
plt.grid()

# Show the plot
plt.show()
'''


# Compute the proximity matrix using the apply method ***************************

def proximityMatrix(model, X, normalize=True):
    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:, 0]
    proxMat = 1 * np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:, i]
        proxMat += 1 * np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat


# Calculate the normalized proximity matrix

# Number of iterations for creating subsets and proximity matrices
num_iterations = 5
subset_fraction = 0.3

# List to store proximity matrices from each iteration
prox_matrices = []

for _ in range(num_iterations):
    # Randomly select a subset of X_train
    num_samples = int(len(X_train) * subset_fraction)
    X_train_subset = resample(X_train, n_samples=num_samples, random_state=42)

    # Calculate proximity matrix for the current subset
    prox_matrix = proximityMatrix(model, X_train_subset, normalize=True)

    # Append the proximity matrix to the list
    prox_matrices.append(prox_matrix)

# Calculate the median proximity matrix
median_prox_matrix = np.median(np.array(prox_matrices), axis=0)

# Convert the median proximity matrix to a dissimilarity matrix
median_dissimilarity_matrix = 1 - median_prox_matrix

# Perform MDS and plot
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
mds_coords = mds.fit_transform(median_dissimilarity_matrix)

# Get the sales values corresponding to the samples in X_train_subset
subset_sales = y_train.iloc[X_train_subset.index]

# Create a scatter plot of MDS coordinates with subset_sales as the color data
plt.scatter(mds_coords[:, 0], mds_coords[:, 1], c=subset_sales, cmap='coolwarm')
plt.colorbar(label='Sales')
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.title("MDS Plot using Median Proximities")
plt.show()