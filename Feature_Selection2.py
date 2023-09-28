import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot, pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
# from Data_Model import market_agg
import xgboost as xgb
# from Feature_Selection import previous_feature_adder, impute_previous_target


# from Feature_Selection import selected_features_v1
def previous_feature_adder(data_set, year_):
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
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    X_feature_ = X_feature_.copy()

    for index_, row_ in X_feature_.iterrows():
        if row_.isnull().any():
            pc_ = row_['Program Code']
            pre_ = data_set[data_set['Program Code'] == pc_]['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
            X_feature_.loc[X_feature_['Program Code'] == pc_, 'Previous_Target'] = pre_

    return X_feature_


# Load data :
# Define the file path
file_path_market_agg = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\market_agg.csv"

# Read the CSV file into a DataFrame
market_agg = pd.read_csv(file_path_market_agg)

selected_features_v1 = np.array(['Total Households Serviced', 'Single Family Dwellings', 'Full User Pay',
                                 'Bag Limit Program for Garbage Collection',
                                 'Municipal Group', 'Single Stream', 'Residential Promotion & Education Costs',
                                 'Program efficiency', 'Interest on Municipal  Capital',
                                 'Total Gross Revenue', 'Interaction of Households Serviced and operation cost',
                                 'operation cost', 'Previous_Target']
                                )
# Columns to add
columns_to_add = np.array(['Year', 'Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes'])

# Append the new columns to selected_features_v1
selected_features_v1_extend = np.append(selected_features_v1, columns_to_add)

# Use loc to select columns based on the first version of feature selection
selected_features_df = market_agg.loc[:, market_agg.columns.isin(selected_features_v1_extend)]


# Helper function to create shadow features
def _create_shadow(x):
    x_shadow = x.copy()
    for c in x_shadow.columns:
        np.random.shuffle(x_shadow[c].values)
    shadow_names = ["shadow_feature_" + str(i + 1) for i in range(x.shape[1])]
    x_shadow.columns = shadow_names
    x_new = pd.concat([x, x_shadow], axis=1)
    return x_new, shadow_names


# Function to evaluate feature importance and select features
def boruta_feature_selection(X_feature, y_target, model, num_iterations=20):
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    X_feature = X_feature.copy()

    # Impute based on the average for NaN values
    X_feature = impute_previous_target(X_feature, market_agg)

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


# Best Hyperparameters resulted form Hyperparameter tuning
best_params = {'bootstrap': True, 'max_depth': 90, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 489}

# Create a regression model
model = RandomForestRegressor(**best_params)

# Get unique years from the data
unique_years = market_agg['Year'].unique()

# Create an empty dictionary to store selected features for each year
selected_features_dict = {}

# Iterate through each year
for year in unique_years:
    if year < 2021:  # Condition to limit the loop
        # Add previous year target and operation cost as additional features
        selected_features_df = selected_features_df.copy()
        X_feature, y_target = previous_feature_adder(selected_features_df, year)

        # Reset the index of X before each iteration
        X_feature.reset_index(drop=True)

        # Reset the index of y_target before each iteration
        y_target.reset_index(drop=True)

        # Create a regression model
        model = RandomForestRegressor(**best_params)

        # Call the function to evaluate feature importance and select features
        boruta_feature_selection(X_feature, y_target, model)

'''
Total Hits Over Iterations:
                                                   var  total hits in iteration
0                           Total Households Serviced                     20.0
1                             Single Family Dwellings                     18.0
2                                       Full User Pay                      0.0
3            Bag Limit Program for Garbage Collection                      0.0
4                                     Municipal Group                     20.0
5                                       Single Stream                      0.0
6             Residential Promotion & Education Costs                     20.0
7                      Interest on Municipal  Capital                     19.0
8                                 Total Gross Revenue                     20.0
9   Interaction of Households Serviced and operati...                     20.0
10                                     operation cost                     20.0
11                                 Program efficiency                      6.0
12                                    Previous_Target                     20.0
'''