# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict
import seaborn as sns

# Create a dictionary with the data
data = {
    'index': list(range(30)),
    'YearsExperience': [1.2, 1.4, 1.6, 2.1, 2.3, 3.0, 3.1, 3.3, 3.3, 3.8, 4.0, 4.1, 4.1, 4.2, 4.6, 5.0, 5.2, 5.4, 6.0,
                        6.1, 6.9, 7.2, 8.0, 8.3, 8.8, 9.1, 9.6, 9.7, 10.4, 10.6],
    'Salary': [39344, 46206, 37732, 43526, 39892, 56643, 60151, 54446, 64446, 57190, 63219, 55795, 56958, 57082, 61112,
               67939, 66030, 83089, 81364, 93941, 91739, 98274, 101303, 113813, 109432, 105583, 116970, 112636, 122392,
               121873]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Splitting the Dataset into Train & Test Dataset
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regressor.fit(X_train, y_train)

# Training Accuracy
y_pred_train = rf_regressor.predict(X_train)
train_accuracy = r2_score(y_train, y_pred_train)

# Testing Accuracy
y_pred = rf_regressor.predict(X_test)
test_accuracy = r2_score(y_test, y_pred)

# Improving Results with K Cross Validation & Hyperparameter Tuning
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, None],
    'max_features': ["sqrt", "log2", None],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [10, 25, 50, 75, 100]
}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Checking for Best Hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Training Accuracy with Best Hyperparameters
best_grid = grid_search.best_estimator_
y_pred_train_best = best_grid.predict(X_train)
train_accuracy_best = r2_score(y_train, y_pred_train_best)
print("Training Accuracy:", train_accuracy_best)

# Evaluate performance with Best Hyperparameters
y_pred_best = best_grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred_best)
print(f"Mean Squared Error: {mse}")
# print("Testing Accuracy:", r2)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred_best)
rmse = mean_squared_error(y_test, y_pred_best, squared=False)
mae = mean_absolute_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")
print("-_-" * 27)

# Create a DataFrame for actual and predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_best})

# Create a scatter plot
plt.scatter(x=results.index, y=results['Actual'], label='Actual', color='blue')
plt.scatter(x=results.index, y=results['Predicted'], label='Predicted', color='orange')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

# OOB error for the best_params
oob_error = 1 - best_grid.oob_score
print("Out-of-Bag Error:", oob_error)

# Create a figure object and set the size
plt.figure(figsize=(10, 6))

# Range of `n_estimators` values to explore
min_estimators = 15
max_estimators = 150

# Define a list of ensemble classifiers with their labels
ensemble_clfs = [("Random Forest (10 max_depth)", RandomForestRegressor(max_depth=10, oob_score=True, random_state=42))]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1, 5):
        clf.set_params(n_estimators=i)
        clf.fit(X_train, y_train)

        # Record the OOB error for each `n_estimators=i` setting
        oob_error = 1 - clf.oob_score_
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

# Compute the proximity matrix using the apply method


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
prox_matrix = proximityMatrix(rf_regressor, X_train, normalize=True)
print("Proximity matrix dimensions:", prox_matrix.shape)
print(prox_matrix)

# Compute the dissimilarity matrix from the proximity matrix
dissimilarity_matrix = 1 - prox_matrix

# Perform MDS and plot
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
mds_coords = mds.fit_transform(dissimilarity_matrix)

# Create a scatter plot of MDS coordinates
plt.scatter(mds_coords[:, 0], mds_coords[:, 1], c=y_train, cmap='coolwarm')
plt.colorbar(label='Salary')
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.title("MDS Plot using Random Forest Proximities")
plt.show()
