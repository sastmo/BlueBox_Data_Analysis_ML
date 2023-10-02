"""
These scripts provide a comprehensive framework for random feature selection.
It generates a random value for each column based on the column data type and visualizes the feature importance curve
for real and random features. It also drops those features which have a lower score than their random counterparts.

Additionally, this code goes even further by evaluating feature importance multiple times
(controlled by `num_iterations`) to account for variability caused by randomness and statistics.
It calculates the number of times each feature passes the importance threshold and considers them as
important features, enhancing the robustness of feature selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from matplotlib.lines import Line2D

# Load data :
# Define the file path
file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\result_classification.csv"

# Read the CSV file into a DataFrame
recycle_material = pd.read_csv(file_path)


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


# Function to add the previous year's target as a feature
def add_previous_year_target(data_set, year_):
    """
    Adds the previous year's target as a feature to the data.

    Args:
        data_set (DataFrame): Input data.
        year_ (int): Year for which to add the previous year's target.

    Returns:
        DataFrame: Data with the 'Previous_Target' feature added.
    """
    if year_ == 2019:
        # For the initial year (2019), use the mean of the target from the same Program Code
        year_data = data_set[data_set['Year'] == year_]

        # Calculate and add the previous year's target as a feature
        previous_year_avg = data_set.groupby('Program Code')['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
        year_data['Previous_Target'] = year_data['Program Code'].map(previous_year_avg)

        return year_data
    else:
        # For subsequent years, use the previous year's target
        year_data = data_set[data_set['Year'] == year_]

        # Calculate and add the previous year's target as a feature
        previous_year = year_ - 1
        previous_feature = data_set[data_set['Year'] == previous_year][
            ['Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes']]

        # Create a mapping of 'Program Code' to target for the previous year
        previous_mapping = previous_feature.set_index('Program Code')[
            'TOTAL Reported and/or Calculated Marketed Tonnes'].to_dict()

        # Use the mapping to create the 'Previous_Target' column in the data set
        year_data['Previous_Target'] = year_data['Program Code'].map(previous_mapping)

        return year_data


# Function to impute missing previous target values
def impute_previous_target(X_feature_, data_set):
    """
    Imputes missing previous target values by taking the mean of the target for the same Program Code.

    Args:
        data_set (DataFrame): Input data with 'Previous_Target' feature.

    Returns:
        X_feature_
    """
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    X_feature_ = X_feature_.copy()

    for index_, row_ in X_feature_.iterrows():
        if row_.isnull().any():
            pc_ = row_['Program Code']
            pre_ = data_set[data_set['Program Code'] == pc_]['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
            X_feature_.loc[X_feature_['Program Code'] == pc_, 'Previous_Target'] = pre_

    return X_feature_


# Generate random features based on the data type of existing columns in the DataFrame X
def generate_random_features(X):
    """
    Generate Random Features for Given DataFrame.

    This function generates random features based on the data type of existing columns in the DataFrame X.

    Parameters:
    X (DataFrame): Input DataFrame containing the original features.

    Returns:
    DataFrame: A new DataFrame with random features added.
    """
    random_features = pd.DataFrame()
    for column in X.columns:
        if X[column].dtype == bool:
            random_feature = np.random.choice([0, 1], size=len(X))
        elif X[column].dtype == float:
            random_feature = np.random.uniform(0, 1, size=len(X))
        else:
            random_feature = np.random.randint(0, 100, size=len(X))
        random_features[column + "_random"] = random_feature

    combine = pd.concat([X, random_features], axis=1)

    return pd.concat([X, random_features], axis=1)


# Function to evaluate and plot feature importance and select features that passed the importance threshold
def evaluate_feature_importance_addRandomColumn(X_feature, y_target, model):
    """
    Evaluate and plot feature importance scores for real features and random features.
    Drop features that received less score than their random counterparts.

    Parameters:
    X_feature (DataFrame): Input DataFrame containing the features.
    y_target (Series): Target variable.
    model: Machine learning model for feature importance evaluation.

    Returns:
    list: A list of selected features that passed the importance threshold.
    """

    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    X_feature = X_feature.copy()

    # Drop the 'Year' column from X_feature
    X_feature = X_feature.drop(columns=['Year'])

    # Impute NaN values based on the average of previous targets
    X_feature = impute_previous_target(X_feature, recycle_material)

    # Reset the index of X_feature before each iteration
    X_feature.reset_index(drop=True)

    # Reset the index of y_target before each iteration
    y_target.reset_index(drop=True)

    # Feature Importance = Original Features + Random Features Method
    combined_OriginalRandom_features = generate_random_features(X_feature)

    # Prepare the training features and target
    X_train = combined_OriginalRandom_features
    y_train = y_target

    # Create an instance of StandardScaler
    scaler = StandardScaler()

    # Fit and transform the training data using the scaler
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Get feature importances
    feature_importances = model.feature_importances_

    # Plot feature importances
    importance_df = pd.DataFrame(
        {'Feature': combined_OriginalRandom_features.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by=['Feature', 'Importance'], ascending=[False, False])

    # Assign colors based on feature type (original or random)
    colors = ['blue' if col in X_feature.columns else 'orange' for col in importance_df['Feature']]

    # Plot original and random features side by side with different colors
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette=colors)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')

    # Create a legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Original'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Random')]
    plt.legend(handles=legend_elements, title='Feature Type')

    plt.show()

    # Create a list to store selected features
    selected_features = []

    # Compare feature importances with random features
    num_original_features = len(X_feature.columns)
    for i in range(num_original_features):
        original_importance = feature_importances[i]
        random_importance = feature_importances[i + num_original_features]

        if original_importance > random_importance:
            selected_features.append(X_feature.columns[i])

    return selected_features


# Define a function to evaluate feature importance based on a specified model
def evaluate_feature_importance_with_frequency(X_feature, y_target, model, feature_importanceModel, num_iterations=20):
    """
    Feature Importance Evaluation with Frequency

    This function evaluates feature importance multiple times (controlled by `num_iterations`)
    to account for variability caused by randomness and statistics. It calculates the number of times each
    feature passes the importance threshold and considers them as important features, enhancing the robustness
    of feature selection.

    Parameters:
    - X_feature (DataFrame): Input DataFrame containing the features.
    - y_target (Series): Target variable.
    - model: Machine learning model for feature importance evaluation.
    - feature_importanceModel: A function for feature importance evaluation.
    - num_iterations (int): Number of iterations to evaluate feature importance.

    Returns:
    - list: A list of selected features based on frequency of passing the importance threshold.
    """

    selected_features_frequency = []

    # Loop through iterations for feature importance evaluation
    for _ in range(num_iterations):
        selected_features_ = feature_importanceModel(X_feature, y_target, model)
        selected_features_frequency.append(selected_features_)

    # Create a frequency dictionary to count the frequency of each selected feature
    feature_frequency = {}
    for iteration in selected_features_frequency:
        for feature in iteration:
            if feature in feature_frequency:
                feature_frequency[feature] += 1
            else:
                feature_frequency[feature] = 1

    # Sort the features by frequency
    sorted_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)

    # Separate features and their frequencies
    features, frequencies = zip(*sorted_features)

    # Calculate and plot the probability mass function using binomial distribution
    trials = num_iterations
    pmf = [sp.stats.binom.pmf(x, trials, 0.5) for x in range(trials + 1)]

    # Plot the probability mass function
    plt.plot(list(range(0, trials + 1)), pmf, color="black")

    # Visualize hits for each feature
    displayed_features = set()  # To keep track of displayed features
    label_shift = 0.2  # Adjust this value for spacing between labels
    vertical_spacing = 0.015  # Adjust this value for vertical spacing between labels

    for feature, frequency in zip(features, frequencies):
        color = 'green' if frequency > num_iterations / 2 else 'red'
        x_position = frequency - 1.5
        y_position = 0.002

        # Ensure the labels don't overlap vertically
        while y_position in [y for _, y in displayed_features]:
            y_position += vertical_spacing

        # Add vertical line and label for each feature
        pyplot.axvline(x_position, color=color)
        pyplot.text(x_position, y_position, feature)
        displayed_features.add((feature, y_position))  # Keep track of displayed features and their positions

    # Create a legend
    legend_labels = ['Highly Significant', 'Not Significant']
    legend_colors = ['green', 'red']
    legend_patches = [pyplot.Line2D([0], [0], color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    pyplot.legend(handles=legend_patches, loc='upper right')

    # Show the plot
    pyplot.show()

    return features


# Best Hyperparameters resulted form Hyperparameter tuning
best_params = {'bootstrap': True, 'max_depth': 80, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 400}

# Get unique years from the data
unique_years = recycle_material['Year'].unique()

# Create an empty dictionary to store selected features for each year
selected_features_dict = {}

# Iterate through each year
# Iterate through each year
for year in unique_years:
    if year == 2020:  # Condition to limit the loop
        # Add previous year target and operation cost as additional features
        X_feature, y_target = feature_target_selection(recycle_material, year)

        # Create a regression model
        model = RandomForestRegressor(**best_params)

        # Call the function to evaluate feature importance and select features
        selected_features = evaluate_feature_importance_addRandomColumn(X_feature, y_target, model)

        # evaluate feature importance based on frequency
        evaluate_feature_importance_with_frequency(X_feature, y_target, model, evaluate_feature_importance_addRandomColumn)



