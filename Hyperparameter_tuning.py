# https://chat.openai.com/share/4fdf92be-5a23-4ca7-b387-557b29cdd20e
# https://medium.com/@ali.soleymani.co/stop-using-grid-search-or-random-search-for-hyperparameter-tuning-c2468a2ff887

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Create a dictionary with the data (same as before)
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

# Random Hyperparameter Grid
n_estimators = [int(x) for x in np.linspace(start=200, stop=500, num=20)]
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

# Create a base model
rf = RandomForestRegressor(random_state=42)

# Randomized Search
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the random search model
rf_random.fit(X_train, y_train)

# Get the best hyperparameters from random search
best_random = rf_random.best_estimator_

# Update the parameter ranges for grid search based on random search results
param_grid = {
    'n_estimators': [best_random.n_estimators - 100, best_random.n_estimators, best_random.n_estimators + 100],
    'max_depth': [best_random.max_depth - 10, best_random.max_depth, best_random.max_depth + 10],
    'min_samples_split': [best_random.min_samples_split - 2, best_random.min_samples_split,
                          best_random.min_samples_split + 2],
    'min_samples_leaf': [best_random.min_samples_leaf - 1, best_random.min_samples_leaf,
                         best_random.min_samples_leaf + 1],
    'bootstrap': [True]
}

# Instantiate the grid search model
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Fit the grid search model
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from grid search
best_params = grid_search.best_params_

# Create a new RandomForestRegressor instance with the best hyperparameters
best_rf = RandomForestRegressor(**best_params)

# Fit the best_rf model
best_rf.fit(X_train, y_train)


# Evaluate the best_rf model
def evaluate(model, features, labels):
    predictions = model.predict(features)
    mae = mean_absolute_error(labels, predictions)
    accuracy = 100 * (1 - mae / np.mean(labels))
    return accuracy


best_accuracy = evaluate(best_rf, X_test, y_test)
print("Best Model Accuracy:", best_accuracy)

# Evaluate other models for comparison
base_rf = RandomForestRegressor(random_state=42)
base_rf.fit(X_train, y_train)
base_accuracy = evaluate(base_rf, X_test, y_test)

random_accuracy = evaluate(best_random, X_test, y_test)

print("Base Model Accuracy:", base_accuracy)
print("Random Search Model Accuracy:", random_accuracy)
print("Best Hyperparameters:", best_params)



