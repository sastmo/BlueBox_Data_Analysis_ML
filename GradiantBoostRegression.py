"""
This script explores recycling trends using Gradient Boosting Regression. It loads and manipulates data,
engineers features, and employs the Gradient Boosting Regression model for prediction.
Hyperparameter tuning, learning curve,early stopping, and feature importance analysis optimize model performance.
Partial dependence plots reveal feature relationships.
This script provides a holistic approach to understanding and predicting recycling behaviors for a sustainable future.
"""

import pandas as pd
from GradiantBoostingClass import GradientBoostingModel


# Function to load and manipulate data
def load_and_manipulate_data(file_path, selected_features_v2):
    """
    Load data from a CSV file, manipulate it, and return the selected features.

    Args:
    - file_path (str): Path to the CSV file.
    - selected_features_v2 (list): List of feature names to select.

    Returns:
    - selected_features (DataFrame): DataFrame containing the selected features.
    """

    # Read the CSV file into a DataFrame
    clustered_data = pd.read_csv(file_path)

    # Append the new columns to selected_features_v1
    columns_to_add = ['Year', 'Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes']
    selected_features_v2_extend = selected_features_v2 + columns_to_add

    # Select columns based on the first version of feature selection
    data_set = clustered_data.loc[:, clustered_data.columns.isin(selected_features_v2_extend)]

    return data_set


# Function to add the previous year's target as a feature
def feature_target_selection(data_set, year_):
    """
    Selects features and target for a specific year from the dataset.

    Parameters:
    - data_set (DataFrame): The dataset containing recycling material data.
    - year_ (int): The year for which features and target are selected.

    Returns:
    - X_feature_ (DataFrame): Features for the specified year.
    - y_target_ (Series): Target variable for the specified year.
    """
    year_data = data_set[data_set['Year'] == year_]

    # Select columns before the target column
    y_target_ = year_data['TOTAL Reported and/or Calculated Marketed Tonnes']
    X_feature_ = year_data.drop(columns=['TOTAL Reported and/or Calculated Marketed Tonnes'])

    return X_feature_, y_target_


# Load data :
# Define the file path
file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\result_classification.csv"

# Define the selected features list
selected_features_v2 = [
    'Total Households Serviced', 'Single Family Dwellings', 'Full User Pay', 'Municipal Group',
    'Bag Limit Program for Garbage Collection', 'Interaction of Households Serviced and operation cost',
    'Residential Promotion & Education Costs', 'Interest on Municipal Capital',
    'Total Gross Revenue', 'Previous_Target', 'operation cost', 'Program efficiency'
]

# Call the function to load and manipulate the data
recycle_material = load_and_manipulate_data(file_path, selected_features_v2)

# Get unique years from the data
unique_years = recycle_material['Year'].unique()

# Iterate through each year
for year in unique_years:
    if year == 2020:  # Condition to limit the loop
        # Add previous year target and operation cost as additional features
        X_feature, y_target = feature_target_selection(recycle_material, year)

        # Reset the index of X before each iteration
        X_feature.reset_index(drop=True)

        # Reset the index of y_target before each iteration
        y_target.reset_index(drop=True)

        # Prepare the training features and target
        X_train = X_feature.drop(['Year', 'Program Code'], axis=1).copy()
        y_train = y_target

        # Prepare the test features for prediction (test is the next year)
        next_year = year + 1
        X_feature_test, y_target_test = feature_target_selection(recycle_material, next_year)

        # Reset the index of X before each iteration
        X_feature_test.reset_index(drop=True)

        # Reset the index of y_target before each iteration
        y_target_test.reset_index(drop=True)

        # Prepare the test features and target
        X_test = X_feature_test.drop(['Year', 'Program Code'], axis=1).copy()
        y_test = y_target_test

        # Create an instance of the GradientBoostingModel class
        gb_model = GradientBoostingModel(random_seed=0)

        gb_model.evaluate(X_train, y_train, X_test, y_test)

        # Fit the model and perform the Randomized Search for Hyperparameter tuning
        gb_model.tuning_learning_curve(X_train, y_train, X_test, y_test)

        # Revise the Hyperparameter tuning and Feature Importance Analysis
        gb_model.perform_search_early_stopping(X_train, y_train, X_test, y_test, X_train.columns)

        # Define the initial features for partial dependence
        initial_features_1 = ['Previous_Target', 'Total Households Serviced', 'Program efficiency']

        initial_features_2 = ['operation cost', 'Interaction of Households Serviced and operation cost',
                              ('Total Gross Revenue', 'Program efficiency')]

        # Generate and plot partial dependence
        gb_model.plot_partial_dependence(X_train, initial_features_1)
        gb_model.plot_partial_dependence(X_train, initial_features_2)

'''
1. First iteration of tuning
The best hyperparameter using Random Search: {'subsample': 1.0, 'n_estimators': 1000, 'min_samples_leaf': 5, 
'max_features': 1.0, 'max_depth': 6, 'loss': 'huber', 'learning_rate': 0.1, 'alpha': 0.9}

Huber Loss (train): 259.2333
Huber Loss (test): 292.5908
R^2 Score (train): 0.9072
R^2 Score (test): 0.9354
'''

'''
2. Second iteration of tuning using early_stopping
The best hyperparameter using Random Search: {'subsample': 1.0, 'min_samples_leaf': 3, 'max_features': 1.0, 
'max_depth': 5, 'loss': 'huber', 'learning_rate': 0.05, 'alpha': 0.9}
Number of tree estimator based on early stopping method is 150
Huber Loss (train): 195.5190
Huber Loss (test): 311.8101
R^2 Score (train): 0.9659
R^2 Score (test): 0.9761
'''
