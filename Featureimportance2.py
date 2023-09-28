# https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/feature_importance.ipynb
# https://chat.openai.com/share/f7b3e512-60af-4c1b-9552-ad2e1cd7442d

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from rfpimp import importances, plot_importances
from sklearn.ensemble import RandomForestRegressor

# Read the CSV file with a two-space delimiter
file_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\housing.csv"
boston = pd.read_csv(file_path, sep=r'\s+', na_values=["nall"], engine='python')

error_ = []
for i, row in enumerate(boston.values):
    if len(row) == 14:
        error_.append(i)

print(len(error_), len(boston), "\n-->")

# Define the column names
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
    'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# Add the column names to the DataFrame
boston.columns = column_names

# Display the modified DataFrame with column names
print(boston.head())

# Separate target variable 'MEDV' and features
y = boston['MEDV']
X = boston.drop('MEDV', axis=1)  # Removed unnecessary quotation mark

# Adding a random column
np.random.seed(seed=42)
X['random'] = np.random.random(size=len(X))

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# Plot y vs random
plt.figure(figsize=(10, 6))
plt.scatter(X_train['random'], y_train, alpha=0.5)
plt.title('Scatter Plot: y vs Random Feature')
plt.xlabel('Random Feature')
plt.ylabel('y')
plt.show()

# Combine the target variable with features for correlation matrix
X_with_target = pd.concat([X, y], axis=1)

# Calculate the correlation matrix
correlation_matrix = X_with_target.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Create a RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,
                           n_jobs=-1,
                           oob_score=True,
                           bootstrap=True,
                           random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions on the training and validation data
y_train_pred = rf.predict(X_train)
y_valid_pred = rf.predict(X_valid)

# Calculate R^2 scores
r2_train = r2_score(y_train, y_train_pred)
r2_valid = r2_score(y_valid, y_valid_pred)

# Get the OOB score
oob_score = rf.oob_score_

print('R^2 Training Score:', r2_train)
print('R^2 Validation Score:', r2_valid)
print('OOB Score:', oob_score)

'''# Create a Ridge regression model
ridge = Ridge(alpha=1.0)

# Fit the model to the training data
ridge.fit(X_train, y_train)

# Make predictions on the validation data
y_valid_pred = ridge.predict(X_valid)

# Calculate the R^2 score on the validation data
r2_valid = r2_score(y_valid, y_valid_pred)

print('Ridge R^2 Validation Score:', r2_valid)

# Create an Elastic Net model
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

# Fit the model to the training data
elastic_net.fit(X_train, y_train)

# Make predictions on the validation data
y_pred = elastic_net.predict(X_valid)

# Calculate the R^2 score on the validation data
r2_ = r2_score(y_valid, y_pred)

print('Elastic Net R^2 Validation Score:', r2_)

# 1. Overall feature importances
# 1.1.Default Scikit-learn’s feature importances

importance = rf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importance)[::-1]

# Plot the sorted feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importance[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)  # Use column names for x-axis labels
plt.tight_layout()
plt.show()

# 1.2. Permutation Feature Importance:

# Calculate feature importance's using permutation_importance
perm_importance = permutation_importance(rf, X_valid, y_valid, n_repeats=30, random_state=42)

# Get feature names and importance's
feature_names = X_valid.columns
importances_mean = perm_importance.importances_mean
importances_std = perm_importance.importances_std

# Sort feature importance's
indices = np.argsort(importances_mean)[::-1]

# Plot the sorted feature importance's
plt.figure(figsize=(10, 6))
plt.title("Permutation Feature Importance's")
plt.bar(range(X_valid.shape[1]), importances_mean[indices], yerr=importances_std[indices], align="center")
plt.xticks(range(X_valid.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

# Calculate feature importances using rfpimp

importances = importances(rf, X_valid, y_valid)

# Set the figure size
plt.figure(figsize=(10, 6))

# Plot the feature importances using rfpimp's plot_importances
plot_importances(importances)
plt.title("Feature Importances (rfpimp)")
plt.show()

# 1.3. Drop Column feature importance

from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt

def drop_col_feat_imp(model, X_train, y_train, random_state=42):
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []

    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis=1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis=1), y_train)
        importances.append(benchmark_score - drop_col_score)

    importances_df = imp_df(X_train.columns, importances)
    return importances_df


def imp_df(cols, imp):
    return pd.DataFrame({'Feature': cols, 'Importance': imp})


def plot_importances(importances_df):
    importances_df.sort_values(by='Importance', ascending=False, inplace=True)  # Sort the DataFrame
    plt.figure(figsize=(10, 6))
    plt.barh(importances_df['Feature'], importances_df['Importance'], align='center')
    plt.xlabel('Improvement in R²')
    plt.title('Drop Column Feature Importances')
    plt.show()


# Usage example
importances_df = drop_col_feat_imp(rf, X_train, y_train)
importances_df_sorted = importances_df.sort_values(by='Importance', ascending=False)  # Sort the DataFrame
print(importances_df_sorted)

# Plot feature importances
plot_importances(importances_df_sorted)

# uses the out-of-bag (OOB) error for evaluating feature importance using the model.oob_score
# using the OOB error for feature importance provides a more realistic estimate of how dropping a
# particular feature affects the model's performance. It takes into account the ensemble nature
# of the Random Forest and ensures that the evaluation is done on unseen data,
# improving the reliability of the feature importance estimates.

from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt


def drop_col_feat_imp(model, X_train, y_train, random_state=42):
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.oob_score_
    # list for storing feature importances
    importances = []

    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis=1), y_train)
        drop_col_score = model_clone.oob_score_
        importances.append(benchmark_score - drop_col_score)

    importances_df = imp_df(X_train.columns, importances)
    return importances_df


def imp_df(cols, imp):
    return pd.DataFrame({'Feature': cols, 'Importance': imp})


def plot_importances(importances_df):
    importances_df.sort_values(by='Importance', ascending=False, inplace=True)  # Sort the DataFrame
    plt.figure(figsize=(10, 6))
    plt.barh(importances_df['Feature'], importances_df['Importance'], align='center')
    plt.xlabel('Improvement in OOB Score')
    plt.title('Drop Column Feature Importances using OOB Score')
    plt.show()


# Usage example
importances_df = drop_col_feat_imp(rf, X_train, y_train)
importances_df_sorted = importances_df.sort_values(by='Importance', ascending=False)  # Sort the DataFrame
print(importances_df_sorted)

# Plot feature importances
plot_importances(importances_df_sorted)
plt.show()
'''

# 2. Observation Level Feature Importance:

# 2.1. Treeinterpreter

from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Create a RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get predictions and contributions using treeinterpreter
prediction, bias, contributions = ti.predict(rf, X_valid)

# Choose an observation index to analyze (e.g., index 31)
observation_index = 31

# Print prediction and actual value for the chosen observation
print(f"Prediction: {prediction[observation_index][0]:.3f} Actual Value: {y_valid.iloc[observation_index]:.3f}")
print(f"Bias (trainset mean): {bias[0]:.3f}")  # Access the first element of the bias array

# Print feature contributions for the chosen observation
for feature, contribution in zip(X_valid.columns, contributions[observation_index]):
    print(f"{feature}: {contribution:.3f}")

# Plot feature contributions for the chosen observation
plt.figure(figsize=(10, 6))
plt.barh(range(len(X_valid.columns)), contributions[observation_index], align="center")
plt.yticks(range(len(X_valid.columns)), X_valid.columns)
plt.xlabel("Feature Contribution")
plt.title("Feature Contributions for Chosen Observation")
plt.show()

# 2.2.LIME (Local Interpretable Model-agnostic Explanations)

import lime
import lime.lime_tabular
import numpy as np

# Create a LimeTabularExplainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   mode='regression',
                                                   feature_names=X_train.columns,
                                                   categorical_features=[3],
                                                   categorical_names=['CHAS'],
                                                   discretize_continuous=True)

# Explain the first observation (Index: 31)
exp_31 = explainer.explain_instance(X_train.values[31], rf.predict, num_features=5)
explanation_text_31 = exp_31.as_list()  # Get the explanation as a list of tuples
explanation_plot_31 = exp_31.as_pyplot_figure()  # Get the explanation as a pyplot figure

# Explain the second observation (Index: 85)
exp_85 = explainer.explain_instance(X_train.values[85], rf.predict, num_features=5)
explanation_text_85 = exp_85.as_list()  # Get the explanation as a list of tuples
explanation_plot_85 = exp_85.as_pyplot_figure()  # Get the explanation as a pyplot figure

print("Explanation for Observation 31:")
print(explanation_text_31)
explanation_plot_31.show()

print("\nExplanation for Observation 85:")
print(explanation_text_85)
explanation_plot_85.show()
