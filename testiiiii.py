import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from GradiantBoostingClass import GradientBoostingModel


# from GradiantBoostingClass import GradientBoostingModel
# from Data_Model import market_agg
# from Hyperparameter_tune import best_params


def load_and_manipulate_data(file_path, selected_features_v2):
    # Read the CSV file into a DataFrame
    clustered_data = pd.read_csv(file_path)

    # Append the new columns to selected_features_v1
    columns_to_add = ['Year', 'Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes']
    selected_features_v2_extend = selected_features_v2 + columns_to_add

    # Select columns based on the first version of feature selection
    data_set = clustered_data.loc[:, clustered_data.columns.isin(selected_features_v2_extend)]

    return data_set


# Function to add previous year target as a feature
def feature_target_selection(data_set, year_):
    year_data = data_set[data_set['Year'] == year_]

    # Take the natural logarithm of the 'TOTAL Reported and/or Calculated Marketed Tonnes' column
    year_data['TOTAL Reported and/or Calculated Marketed Tonnes'] = np.log(
        year_data['TOTAL Reported and/or Calculated Marketed Tonnes'])

    # Select columns before the target column
    target_column_index = year_data.columns.get_loc('TOTAL Reported and/or Calculated Marketed Tonnes')
    X_feature_ = year_data.iloc[:, :target_column_index]
    y_target_ = year_data['TOTAL Reported and/or Calculated Marketed Tonnes']

    return X_feature_, y_target_


# Function to get features for the next year
def test_data_selection(data_set, year_):
    next_year_ = year_ + 1
    data_set.dropna(axis=0, inplace=True)
    X_feature_, y_target_ = feature_target_selection(data_set, next_year_)
    return X_feature_, y_target_


# Load data :
# Define the file path
file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\result_classification.csv"

# Define the selected features list
selected_features_v2 = [
    'Total Households Serviced', 'Single Family Dwellings', 'Full User Pay',
    'Bag Limit Program for Garbage Collection', 'Municipal Group',
    'Residential Promotion & Education Costs', 'Interest on Municipal Capital',
    'Total Gross Revenue', 'Previous_Target', 'operation cost', 'Program efficiency'
]

# Call the function to load and manipulate the data
recycle_material = load_and_manipulate_data(file_path, selected_features_v2)

# Get unique years from the data
print(recycle_material.columns)

unique_years = recycle_material['Year'].unique()

# Iterate through each year
for year in unique_years:
    if year < 2021:  # Condition to limit the loop
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
        X_feature_test, y_target_test = test_data_selection(recycle_material, year)

        # Reset the index of X before each iteration
        X_feature_test.reset_index(drop=True)

        # Reset the index of y_target before each iteration
        y_target_test.reset_index(drop=True)

        # Prepare the test features and target
        X_test = X_feature_test.drop(['Year', 'Program Code'], axis=1).copy()
        y_test = y_target_test

        # Create an instance of the GradientBoostingModel class
        gb_model = GradientBoostingModel(random_seed=0)

        # Fit the model and perform the second randomized search
        # gb_model.tuning_learning_curve(X_train, y_train, X_test, y_test)
        gb_model.perform_search_early_stopping(X_train, y_train, X_test, y_test)

        # Plot feature importance
        # gb_model.feature_importance(X_train, X_train.columns)

        # Define the initial features for partial dependence
        initial_features = ['Total Households Serviced', 'Residential Promotion & Education Costs',
                            ('Previous_Target', 'Residential Promotion & Education Costs')]

        # Generate and plot partial dependence
        gb_model.plot_partial_dependence(X_train, initial_features)



