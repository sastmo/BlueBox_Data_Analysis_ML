"""
Boruta Feature Selection on Top of Random Feature Selection

# Main Goal:
This script performs Boruta feature selection, enhancing the reliability of feature selection by iterating
multiple times to identify important features based on shadow comparisons and visualize the results.

# Steps and Functions:
1. Data Loading and Manipulation: Load data, append key columns, and prepare the dataset.
2. Shadow Feature Creation: Generate shadow features by shuffling data columns.
3. Boruta Feature Selection with Iteration: Iteratively evaluate feature importances and accumulate hits.
4. Visualization: Plot results and categorize features as "Highly Significant" or "Not Significant."

This script aids in optimal feature selection for machine learning tasks.
"""

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot, pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# Function to load and manipulate data
def load_and_manipulate_data(file_path, selected_features):
    """
    Load and manipulate data from a CSV file.

    This function reads the CSV file located at 'file_path' into a DataFrame, appends new columns
    ('Year', 'Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes') to 'selected_features',
    and selects columns based on the updated 'selected_features'.

    Parameters:
    file_path (str): Path to the CSV file containing the data.
    selected_features (list): List of selected features for analysis.

    Returns:
    DataFrame: Manipulated DataFrame containing selected features and additional columns.
    """
    # Read the CSV file into a DataFrame
    clustered_data = pd.read_csv(file_path)

    # Append the new columns to selected_features_v1
    columns_to_add = ['Year', 'Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes']
    selected_features_v2_extend = selected_features + columns_to_add

    # Select columns based on the first version of feature selection
    data_set = clustered_data.loc[:, clustered_data.columns.isin(selected_features_v2_extend)]

    return data_set


# Function to add the previous year's target as a feature
def previous_feature_adder(data_set, year_):
    """
    Add Previous Year's Target as a Feature

    This function adds the previous year's target as a feature to the dataset based on the specified year.

    Parameters:
    - data_set (DataFrame): Input DataFrame containing the data.
    - year_ (int): The year for which the previous year's target should be added as a feature.

    Returns:
    - X_feature_ (DataFrame): Updated DataFrame with the previous year's target as a feature.
    - y_target_ (Series): Target variable for the specified year.
    """
    if year_ == 2019:
        # For the initial year, use mean of target from the same Program Code
        year_data = data_set[data_set['Year'] == year_]

        # Select columns before the target column
        target_column_index = year_data.columns.get_loc('TOTAL Reported and/or Calculated Marketed Tonnes')
        X_feature_ = year_data.iloc[:, :target_column_index]
        y_target_ = year_data['TOTAL Reported and/or Calculated Marketed Tonnes']

        # Create a copy of the DataFrame to avoid modifying the original DataFrame
        X_feature_ = X_feature_.copy()

        # Calculate and add previous year target as a feature
        pre = data_set.groupby('Program Code')['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
        X_feature_['Previous_Target'] = X_feature_['Program Code'].map(pre)

        return X_feature_, y_target_
    else:
        # For subsequent years, use previous year's target
        year_data = data_set[data_set['Year'] == year_]

        # Select columns before the target column
        target_column_index = year_data.columns.get_loc('TOTAL Reported and/or Calculated Marketed Tonnes')
        X_feature_ = year_data.iloc[:, :target_column_index]
        y_target_ = year_data['TOTAL Reported and/or Calculated Marketed Tonnes']

        # Create a copy of the DataFrame to avoid modifying the original DataFrame
        X_feature_ = X_feature_.copy()

        # Calculate and add previous year target as a feature
        Previous_year = year_ - 1
        Previous_feature = data_set[data_set['Year'] == Previous_year][
            ['Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes']]

        # Create a mapping of 'Program Code' to target for the previous year
        previous_mapping = Previous_feature.set_index('Program Code')[
            'TOTAL Reported and/or Calculated Marketed Tonnes'].to_dict()

        # Use the mapping to create the 'Previous_Target' column in X_feature_
        X_feature_['Previous_Target'] = X_feature_['Program Code'].map(previous_mapping)

        return X_feature_, y_target_


# Function to impute previous target values
def impute_previous_target(X_feature_, data_set):
    """
    Impute Previous Year's Target Values

    This function imputes missing values in the 'Previous_Target' column using the mean target value
    for the corresponding 'Program Code' from the provided dataset.

    Parameters:
    - X_feature_ (DataFrame): Input DataFrame containing the features.
    - data_set (DataFrame): Dataset containing the target variable used for imputation.

    Returns:
    - X_feature_ (DataFrame): Updated DataFrame with imputed 'Previous_Target' values.
    """
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    X_feature_ = X_feature_.copy()

    for index_, row_ in X_feature_.iterrows():
        if row_.isnull().any():
            pc_ = row_['Program Code']
            pre_ = data_set[data_set['Program Code'] == pc_]['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
            X_feature_.loc[X_feature_['Program Code'] == pc_, 'Previous_Target'] = pre_

    return X_feature_


# Helper function to create shadow features
def _create_shadow(x):
    """
    Create Shadow Features for the Given DataFrame.

    This function generates shadow features by shuffling the values of each column in the input DataFrame x.

    Parameters:
    x (DataFrame): Input DataFrame containing original features.

    Returns:
    DataFrame: A new DataFrame with shadow features added.
    list: Names of the shadow features.
    """

    x_shadow = x.copy()

    # Shuffle the values of each column to create shadow features
    for c in x_shadow.columns:
        np.random.shuffle(x_shadow[c].values)

    # Create names for shadow features
    shadow_names = ["shadow_feature_" + str(i + 1) for i in range(x.shape[1])]

    # Rename columns with shadow feature names
    x_shadow.columns = shadow_names
    # Function to evaluate feature importance and select features

    # Concatenate the original features with the shadow features
    x_new = pd.concat([x, x_shadow], axis=1)

    return x_new, shadow_names


# Function to perform Boruta feature selection
def boruta_feature_selection(X_feature, y_target, model, num_iterations=20):
    """
    Boruta Feature Selection Method with Iteration

    This function performs feature selection using the Boruta method with multiple iterations.
    It calculates feature importances for both original features and shadow features created by shuffling data,
    and identifies important features that surpass a shadow threshold.
    It also visualizes the process and results.

    Parameters:
    - X_feature (DataFrame): Input DataFrame containing the features.
    - y_target (Series): Target variable.
    - model: Machine learning model for feature importance evaluation.
    - num_iterations (int): Number of iterations to evaluate feature importance.

    Returns:
    - list: A list of selected features based on the Boruta method.
    """
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    X_feature = X_feature.copy()

    # Impute based on the average for NaN values
    X_feature = impute_previous_target(X_feature, recycle_material)

    print(X_feature[X_feature['Program efficiency'].isna()])

    # Drop the 'Year' column from X_feature
    X_feature = X_feature.drop(columns=['Year', 'Program Code'])

    # Reset the index of X before each iteration
    X_feature_reset_index = X_feature.reset_index(drop=True)

    # Reset the index of y_target before each iteration
    y_target_reset_index = y_target.reset_index(drop=True)

    # Create shadow features using the provided _create_shadow function
    X_boruta, shadow_names = _create_shadow(X_feature_reset_index)
    print("Shadow Features:\n", X_boruta[shadow_names])

    # Create an instance of StandardScaler
    scaler = StandardScaler()

    # Fit and transform the training data using the scaler
    X_boruta_standard = scaler.fit_transform(X_boruta)

    # Fit the specified model
    model.fit(X_boruta_standard, y_target_reset_index)

    # Store and print feature importances
    feat_imp_X = model.feature_importances_[:len(X_feature_reset_index.columns)]
    print("\nFeature VIM = ", feat_imp_X)
    feat_imp_shadow = model.feature_importances_[len(X_feature_reset_index.columns):]
    print("\nShadow VIM = ", feat_imp_shadow)

    # Compute shadow threshold and hits
    shadow_threshold = round(feat_imp_shadow.max(), 3)
    print("\nShadow Threshold = ", shadow_threshold)
    hits = feat_imp_X > shadow_threshold
    print("\nHits = ", hits)

    # Create a DataFrame to accumulate hits over iterations
    hits_counter = np.zeros((len(X_feature_reset_index.columns)))

    # Repeat the process for a specified number of iterations
    for iter_ in range(num_iterations):
        # Create shadow features using the provided _create_shadow function
        X_boruta, shadow_names = _create_shadow(X_feature_reset_index)

        # Create an instance of StandardScaler
        scaler = StandardScaler()

        # Fit and transform the training data using the scaler
        X_boruta_standard = scaler.fit_transform(X_boruta)

        # Fit the specified model
        model.fit(X_boruta_standard, y_target_reset_index)

        # Store feature importance
        feat_imp_X = model.feature_importances_[:len(X_feature_reset_index.columns)]
        feat_imp_shadow = model.feature_importances_[len(X_feature_reset_index.columns):]

        # Calculate hits for this trial and add to the counter
        hits_counter += (feat_imp_X > feat_imp_shadow.max())

        # Print results for each iteration
        print(f"\nIteration {iter_ + 1} Results:")
        print("\nFeature VIM = ", feat_imp_X)
        print("\nShadow VIM = ", feat_imp_shadow)
        print("\nHits = ", hits_counter)

    # Create a DataFrame to display total hits over iterations
    hits_df = pd.DataFrame({'var': X_feature_reset_index.columns, 'total hits in iteration': hits_counter})
    print("\nTotal Hits Over Iterations:\n", hits_df)

    # Calculate and plot the probability mass function using binomial distribution
    trials = num_iterations
    pmf = [sp.stats.binom.pmf(x, trials, 0.5) for x in range(trials + 1)]

    # Plot the probability mass function
    pyplot.plot(list(range(0, trials + 1)), pmf, color="black")

    # Visualize hits for each feature
    displayed_features = set()  # To keep track of displayed features
    label_shift = 0.2  # Adjust this value for spacing between labels
    vertical_spacing = 0.015  # Adjust this value for vertical spacing between labels

    for feature in X_feature_reset_index.columns:
        color = 'green' if hits_df.loc[hits_df[
                                           'var'] == feature, 'total hits in iteration'].values > num_iterations / 2 else 'red'
        x_position = hits_df.loc[hits_df['var'] == feature, 'total hits in iteration'].values - 1.5
        y_position = 0.002

        # Ensure the labels don't overlap vertically
        while y_position in [y for _, y in displayed_features]:
            y_position += vertical_spacing

        pyplot.axvline(x_position, color=color)
        pyplot.text(x_position, y_position, feature)
        displayed_features.add((feature, y_position))  # Keep track of displayed features and their positions

    # Create a legend
    legend_labels = ['Highly Significant', 'Not Significant']
    legend_colors = ['green', 'red']
    legend_patches = [pyplot.Line2D([0], [0], color=color, label=label) for color, label in
                      zip(legend_colors, legend_labels)]
    pyplot.legend(handles=legend_patches, loc='upper right')

    # Show the plot
    pyplot.show()


# Load data :
# Define the file path
file_path_market_agg = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\market_agg.csv"

# List of selected features from the first level of feature selection (Importance + Random Features)
selected_features_v1 = np.array(['Total Households Serviced', 'Single Family Dwellings', 'Full User Pay',
                                 'Bag Limit Program for Garbage Collection',
                                 'Municipal Group', 'Single Stream', 'Residential Promotion & Education Costs',
                                 'Program efficiency', 'Interest on Municipal  Capital',
                                 'Total Gross Revenue', 'Interaction of Households Serviced and operation cost',
                                 'operation cost', 'Previous_Target']
                                )

# Call the function to load and manipulate the data
recycle_material = load_and_manipulate_data(file_path_market_agg, selected_features_v1)


# Best Hyperparameters resulted form Hyperparameter tuning
best_params = {'bootstrap': True, 'max_depth': 80, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 400}

# Get unique years from the data
unique_years = recycle_material['Year'].unique()

# Create an empty dictionary to store selected features for each year
selected_features_dict = {}

# Iterate through each year
for year in unique_years:
    if year < 2021:  # Condition to limit the loop
        # Add previous year target and operation cost as additional features
        selected_features_df = recycle_material.copy()
        X_feature, y_target = previous_feature_adder(recycle_material, year)

        # Reset the index of X before each iteration
        X_feature.reset_index(drop=True)

        # Reset the index of y_target before each iteration
        y_target.reset_index(drop=True)

        # Create a regression model
        model = RandomForestRegressor(**best_params)

        # Call the function to evaluate feature importance and select features
        boruta_feature_selection(X_feature, y_target, model)

