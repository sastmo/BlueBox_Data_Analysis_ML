# https://medium.com/towards-data-science/gradient-boosting-from-theory-to-practice-part-2-25c8b7ca566b
# https://towardsdatascience.com/gradient-boosting-from-theory-to-practice-part-1-940b2c9d8050
# https://chat.openai.com/share/4cb22900-8cc6-4643-8c45-41b57d2bd8c4
# https://www.slideshare.net/PyData/gradient-boosted-regression-trees-in-scikit-learn-gilles-louppe
# https://www.slideshare.net/JaroslawSzymczak1/gradient-boosting-in-practice-a-deep-dive-into-xgboost
# https://app.datacamp.com/workspace/w/ec8aaf23-89b8-474c-b9f2-dcd12bdd8cd5/edit
# https://github.com/roiyeho/medium/tree/main/gradient_boosting

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error as MSE
from Hyperparameter_tuning import n_estimators

# Generate synthetic data
n_samples = 100
X = np.random.rand(n_samples, 1) - 0.5
y = 5 * X[:, 0] ** 2 + 0.1 * np.random.randn(n_samples)

# Visualize the data
plt.scatter(X, y, s=20)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Data')
plt.show()

# First Decision Tree
h1 = DecisionTreeRegressor(max_depth=2)
h1.fit(X, y)

F1 = [h1]  # ensemble of one tree
F1_pred = h1.predict(X)
print(f'R2 score of F1: {r2_score(y, F1_pred):.4f}')

# Second Decision Tree
h2 = DecisionTreeRegressor(max_depth=2)
y2 = y - F1_pred
h2.fit(X, y2)

F2 = [h1, h2]  # ensemble of two trees
F2_pred = sum(h.predict(X) for h in F2)
print(f'R2 score of F2: {r2_score(y, F2_pred):.4f}')

# Third Decision Tree
h3 = DecisionTreeRegressor(max_depth=2)
y3 = y - F2_pred
h3.fit(X, y3)

F3 = [h1, h2, h3]  # ensemble of three trees
F3_pred = sum(h.predict(X) for h in F3)
print(f'R2 score of F3: {r2_score(y, F3_pred):.4f}')

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 7))
X_test = np.linspace(-0.5, 0.5, 500).reshape(-1, 1)

# Plot individual tree predictions and residuals
for i, h, residuals in zip([0, 1, 2], [h1, h2, h3], [y, y2, y3]):
    ax = axes[0, i]
    y_test_pred = h.predict(X_test)
    ax.scatter(X, residuals, c='k', s=20, marker='x', label='Residuals')
    ax.plot(X_test, y_test_pred, 'r', linewidth=2, label='Prediction')
    ax.set_title(f'$h_{i + 1}(x)$')
    ax.legend(loc='upper center')

# Plot ensemble predictions
for i, ensemble in enumerate([F1, F2, F3]):
    ax = axes[1, i]
    y_test_pred = sum(h.predict(X_test) for h in ensemble)
    ax.scatter(X, y, s=20, label='Training set')
    ax.plot(X_test, y_test_pred, 'm', linewidth=2, label='Ensemble Prediction')
    ax.set_title(f'$F_{i + 1}(x)$')
    ax.legend(loc='upper center')

# Set common labels and adjust layout
for ax in axes.flat:
    ax.set_xlabel('X')
    ax.set_ylabel('y')
plt.tight_layout()
plt.show()

# ***************************************************************************************************

# GradientBoostingClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # We only take the first two features
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create and train the Gradient Boosting classifier
clf = GradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_train)

# Print accuracy scores on training and test sets
print(f'Train accuracy: {clf.score(X_train, y_train):.4f}')
print(f'Test accuracy: {clf.score(X_test, y_test):.4f}')

# Define parameter grid for randomized search
params = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': np.arange(3, 11),
    'subsample': np.arange(0.5, 1.0, 0.1),
    'max_features': ['sqrt', 'log2', None]
}

# Perform randomized search with cross-validation
search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), params, n_iter=50, cv=3, n_jobs=-1)
search.fit(X_train, y_train)

# Print best hyperparameters from randomized search
print("Best hyperparameters:", search.best_params_)

# Get the best classifier from the search
best_clf = search.best_estimator_

# Print accuracy scores of the best classifier
print(f'Train accuracy (best): {best_clf.score(X_train, y_train):.4f}')
print(f'Test accuracy (best): {best_clf.score(X_test, y_test):.4f}')

# Set the title for the code
plt.figure(figsize=(10, 6))
plt.title("Gradient Boost Classifier")

# Your plot code can be added here using plt.plot(), plt.scatter(), etc.

# Show the plot
plt.show()

# GradientBoostingRegressor

# Load the housing dataset
file_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\housing.csv"
boston = pd.read_csv(file_path, sep=r'\s+', na_values=["nall"], engine='python')

# Define data types for columns
data_types = {
    'CRIM': float, 'ZN': float, 'INDUS': float, 'CHAS': bool, 'NOX': float,
    'RM': float, 'AGE': float, 'DIS': float, 'RAD': int, 'TAX': int,
    'PTRATIO': float, 'B': float, 'LSTAT': float, 'MEDV': float
}

# Define column names
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
    'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# Assign column names to the DataFrame
boston.columns = column_names

# Separate target variable 'MEDV' and features
y = boston['MEDV']
X = boston.drop('MEDV', axis=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a Gradient Boosting Regressor model
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)

# Calculate and print R2 scores
train_score = reg.score(X_train, y_train)
print(f'R2 score (train): {train_score:.4f}')

test_score = reg.score(X_test, y_test)
print(f'R2 score (test): {test_score:.4f}')

# Define parameter grid for randomized search
params = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': np.arange(3, 11),
    'subsample': np.arange(0.5, 1.0, 0.1),
    'max_features': ['sqrt', 'log2', None]
}

# Perform randomized search with cross-validation
search = RandomizedSearchCV(GradientBoostingRegressor(random_state=0), params, n_iter=50, cv=3, n_jobs=-1)
search.fit(X_train, y_train)

# Print best hyperparameters from randomized search
print("Best hyperparameters:", search.best_params_)

# Get the best regressor from the search
best_reg = search.best_estimator_

# Print R2 scores of the best regressor
print(f'R2 score (train, best): {best_reg.score(X_train, y_train):.4f}')
print(f'R2 score (test, best): {best_reg.score(X_test, y_test):.4f}')

# Scatter plot of predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, reg.predict(X_test), c='blue', label='Baseline Model')
plt.scatter(y_test, best_reg.predict(X_test), c='red', label='Best Model')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values')
plt.legend()
plt.show()

# The Learning Curve

# Calculate test scores for each boosting iteration
test_score = np.zeros(n_estimators)
for i, y_test_pred in enumerate(best_reg.staged_predict(X_test)):
    test_score[i] = MSE(y_test, y_test_pred)

# Plot training and test loss
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(np.arange(n_estimators), best_reg.train_score_, label='Training loss', color='blue')
plt.plot(np.arange(n_estimators), test_score, label='Test loss', color='red')

plt.xlabel('Boosting Iterations')
plt.ylabel('MSE (Mean Squared Error)')
plt.title('Training and Test Loss Curves')
plt.legend()
plt.grid(True)
plt.show()

# Define hyperparameters for randomized search (early stopping)
params = {
    'max_depth': np.arange(3, 11),
    'subsample': np.arange(0.5, 1.0, 0.1),
    'max_features': ['sqrt', 'log2', None]
}

# Perform randomized search with early stopping
search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=0, n_estimators=500, n_iter_no_change=5),
    params, n_iter=50, cv=3, n_jobs=-1
)
search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", search.best_params_)

# Use best estimator to evaluate model performance
reg = search.best_estimator_
print(f'R2 score (train): {reg.score(X_train, y_train):.4f}')
print(f'R2 score (test): {reg.score(X_test, y_test):.4f}')

# Checking number of trees built
print("Number of Trees Built:", reg.n_estimators_)

# Feature Importance

# Sort the features by their importance
feature_importance = best_reg.feature_importances_
sorted_idx = np.argsort(feature_importance)

# Plot the feature importances
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
pos = np.arange(len(feature_importance))

plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Plot')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top

plt.show()
'''
# HistGradientBoostingClassifier Example*********************************************

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
import time

# Create synthetic dataset
X, y = make_hastie_10_2(n_samples=50000, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Using GradientBoostingClassifier
print("GradientBoostingClassifier:")
clf = GradientBoostingClassifier(random_state=0)

start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()

train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(f"Training time: {end_time - start_time:.4f} seconds")
print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Using HistGradientBoostingClassifier
print("\nHistGradientBoostingClassifier:")
clf = HistGradientBoostingClassifier(random_state=0)

start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()

train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(f"Training time: {end_time - start_time:.4f} seconds")
print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

"While most parameters of the histogram-based estimators are similar to the traditional " \
"GradientBoostingClassifier and GradientBoostingRegressor, there are a few differences:" \
"max_iter: Replaces n_estimators and specifies the number of iterations (trees) in the ensemble." \
"Default Tree Size: max_depth is None (no depth limit), max_leaf_nodes is 31, and min_samples_leaf is 20 by default." \
"Automatic Early Stopping: Enabled when the number of samples exceeds 10,000." \
"New Parameters:" \
"max_bins: Specifies the maximum number of bins used for discretization (255 maximum)." \
"categorical_features: List indicating locations of categorical features in the dataset." \
"interaction_cst: Defines interaction constraints among sets of features that can interact in child node splits." \
"Usage and Benefits:" \
"The histogram-based gradient boosting estimators provide a balance between performance and predictive accuracy, " \
"making them highly useful for large datasets." \
"By reducing sorting requirements and leveraging parallelization, these estimators address " \
"computational challenges encountered in traditional gradient boosting methods." \
"Built-in support for handling missing values and categorical features simplifies the preprocessing steps for users."

# XGBoost and LightGBM support for stochastic gradient boosting techniques,************************************

# XGBoost
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XGBoost classifier
params = {
    'objective': 'binary:logistic',
    'subsample': 0.8,               # Sample 80% of the data for each boosting iteration
    'colsample_bytree': 0.8,        # Sample 80% of features for each tree split
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100
}

clf = xgb.XGBClassifier(**params)

# Fit the model
clf.fit(X_train, y_train)

# Evaluate on test data
accuracy = clf.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

# LightGBM

import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM classifier
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'bagging_fraction': 0.8,         # Sample 80% of the data for each boosting iteration
    'feature_fraction': 0.8,         # Sample 80% of features for each tree split
    'max_depth': 3,
    'learning_rate': 0.1,
    'num_iterations': 100
}

clf = lgb.LGBMClassifier(**params)

# Fit the model
clf.fit(X_train, y_train)

# Evaluate on test data
accuracy = clf.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')
'''
