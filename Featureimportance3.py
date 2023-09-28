# https://medium.com/fiverr-engineering/feature-selection-beyond-feature-importance-9b97e5a842f
# https://towardsdatascience.com/feature-selection-how-to-throw-away-95-of-your-data-and-get-95-accuracy-ad41ca016877

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load your housing dataset (replace with the correct file path)
file_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\housing.csv"
boston = pd.read_csv(file_path, sep=r'\s+', na_values=["nall"], engine='python')

# Define the data types for columns
data_types = {
    'CRIM': float, 'ZN': float, 'INDUS': float, 'CHAS': bool, 'NOX': float,
    'RM': float, 'AGE': float, 'DIS': float, 'RAD': int, 'TAX': int,
    'PTRATIO': float, 'B': float, 'LSTAT': float, 'MEDV': float
}

# Define the column names
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
    'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# Add the column names to the DataFrame
boston.columns = column_names

# Separate target variable 'MEDV' and features
y = boston['MEDV']
X = boston.drop('MEDV', axis=1)

'''
# Generate random features based on data types
random_features = pd.DataFrame()
for column in X.columns:
    if X[column].dtype == bool:
        random_feature = np.random.choice([0, 1], size=len(X))
    elif X[column].dtype == float:
        random_feature = np.random.uniform(0, 1, size=len(X))
    else:
        random_feature = np.random.randint(0, 100, size=len(X))
    random_features[column + "_random"] = random_feature

# Combine original and random features
combined_features = pd.concat([X, random_features], axis=1)

print(combined_features.columns)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, y, test_size=0.2, random_state=42)

# Train a random forest regressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

# Get feature importances
feature_importances = regressor.feature_importances_

print(feature_importances)

# Plot feature importances
importance_df = pd.DataFrame({'Feature': combined_features.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by=['Feature', 'Importance'], ascending=[False, False])

# Assign colors based on feature type (original or random)
colors = ['blue' if col in X.columns else 'orange' for col in importance_df['Feature']]

# Plot original and random features side by side with different colors
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette=colors)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')

# Create a legend
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Original'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Random')]
plt.legend(handles=legend_elements, title='Feature Type')

plt.show()

# Create a list to store selected features
selected_features = []

# Compare feature importances with random features
num_original_features = len(X.columns)
for i in range(num_original_features):
    original_importance = feature_importances[i]
    random_importance = feature_importances[i + num_original_features]

    if original_importance > random_importance:
        selected_features.append(X.columns[i])

# Filter X to keep only the selected features
X_filtered = X[selected_features]

print("Selected Features:", selected_features)

# Boruta Approach *****************************************************************

# Set up parameters for Boruta
n_iterations = 5  # Number of iterations for Boruta
perc = 60  # Percentage for Boruta
delta = 0.01  # Delta for stopping criteria


# Helper function to create shadow features
def _create_shadow(x):
    x_shadow = x.copy()
    for c in x_shadow.columns:
        np.random.shuffle(x_shadow[c].values)
    shadow_names = ["shadow_feature_" + str(i + 1) for i in range(x.shape[1])]
    x_shadow.columns = shadow_names
    x_new = pd.concat([x, x_shadow], axis=1)
    return x_new, shadow_names


# Set up the parameters for running the model in XGBoost
param = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    # ... other hyperparameters
}

# Initialize the dataframe to store importance scores
df = pd.DataFrame()

# Loop through each feature in X_filtered
for feature_idx, feature in enumerate(X_filtered.columns):
    # Initialize an array to store importance scores for the current feature
    importance_scores = []

    # Iterate through iterations
    for i in range(1, n_iterations + 1):
        # Create the shadow variables and run the model to obtain importances
        new_x, shadow_names = _create_shadow(X_filtered)
        shadow_column_idx = shadow_names.index(f"shadow_feature_{feature_idx + 1}")

        # Replace the shadow of the current feature
        X_filtered.iloc[:, shadow_column_idx] = new_x.iloc[:, (len(X_filtered.columns) + shadow_column_idx)]

        # Initialize XGBoost model with the given parameters
        bst = xgb.XGBRegressor(**param)
        # Fit the model with the modified X_filtered
        bst.fit(X_filtered, y)

        # Get feature importance for the current feature
        importance_scores.append(bst.feature_importances_[feature_idx])

    # Add importance scores to the DataFrame
    column_name = f'iteration_{n_iterations}'
    df[feature] = importance_scores

# Transpose the DataFrame
df_transposed = df.transpose()

# Pivot the transposed DataFrame
df_pivoted = df_transposed.pivot_table(index=df_transposed.index, values=df_transposed.columns, aggfunc='mean')
print(df_pivoted)

# Calculate mean importance scores across iterations for each feature
df_pivoted['mean_importance'] = df_pivoted.mean(axis=1)

# Calculate the mean shadow importance threshold
mean_shadow = df_pivoted['mean_importance'].mean() * (perc / 100)

# Filter real features based on mean shadow importance
selected_features_boruta = df_pivoted.index[df_pivoted['mean_importance'] > mean_shadow]

# Check stopping criteria
if len(selected_features_boruta) > delta * len(X_filtered.columns):
    selected_feature_names = list(selected_features_boruta)
    print("Selected Features (Boruta):", selected_feature_names)
else:
    print("Stopping criteria not met. Keep more features.")
'''
# Lasso (Least Absolute Shrinkage and Selection Operator) is a linear regression technique

# Use LassoCV for feature selection
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" % reg.score(X, y))
coef = pd.Series(reg.coef_, index=X.columns)

selected_features = X.columns[coef != 0]
print("Selected Features:", selected_features)

print(
    "Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
plt.figure(figsize=(8.0, 10.0))
imp_coef.plot(kind="barh")
plt.title("Feature importance using Lasso Model")
plt.show()
