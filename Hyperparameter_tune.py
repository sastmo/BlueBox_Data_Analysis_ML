import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
# from Data_Model import market_agg
from collections import OrderedDict

# Load data :
# Define the file path
file_path_market_agg = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\market_agg.csv"

# Read the CSV file into a DataFrame
market_agg = pd.read_csv(file_path_market_agg)


# Function to add previous year target as a feature
def previous_feature_adder(data_set, year_):
    delete_after_use = False  # Flag to indicate whether to delete after use

    if year_ == 2019:
        # For the initial year, use mean of target from the same Program Code
        year_data = data_set[data_set['Year'] == year_]

        # Select columns before the target column
        target_column_index = year_data.columns.get_loc('TOTAL Reported and/or Calculated Marketed Tonnes')
        X_feature_ = year_data.iloc[:, :target_column_index]
        y_target_ = year_data['TOTAL Reported and/or Calculated Marketed Tonnes']

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


# Function to get features for the next year
def next_year_data(data_set, year_):
    next_year_ = year_ + 1
    X_feature_, y_target_ = previous_feature_adder(data_set, next_year_)
    return X_feature_, y_target_


# Function to impute previous target values
def impute_previous_target(X_feature_, data_set):
    for index_, row_ in X_feature_.iterrows():
        if row_.isnull().any():
            pc_ = row_['Program Code']
            pre_ = data_set[data_set['Program Code'] == pc_]['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
            X_feature_.loc[X_feature_['Program Code'] == pc_, 'Previous_Target'] = pre_


# Evaluate hyperparameter accuracy to determine the best_rf model
def evaluate_accuracy(model, features, labels):
    predictions = model.predict(features)
    mae = mean_absolute_error(labels, predictions)
    accuracy = 100 * (1 - mae / np.mean(labels))
    return accuracy


# Random search to find the initial range for hyperparameter tuning
def random_search(X_train, y_train, model):
    # Define the hyperparameter grid for Randomized Search
    n_estimators = [int(x) for x in np.linspace(start=100, stop=400, num=20)]
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

    # Randomized Search
    rf_random = RandomizedSearchCV(
        estimator=model,
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

    return best_random


# Grid Search hyperparameter tuning to find the best parameters
def grid_search(X_train, y_train, model, best_random):
    # Define default max_depth in case it's None
    default_max_depth = 10

    # Modify the param_grid based on random search results
    param_grid = {
        'n_estimators': [best_random.n_estimators - 50, best_random.n_estimators, best_random.n_estimators + 50],
        'max_depth': [
            best_random.max_depth - 10 if best_random.max_depth is not None else default_max_depth,
            best_random.max_depth if best_random.max_depth is not None else default_max_depth,
            best_random.max_depth + 10 if best_random.max_depth is not None else default_max_depth
        ],
        'min_samples_split': [best_random.min_samples_split - 2, best_random.min_samples_split,
                              best_random.min_samples_split + 2],
        'min_samples_leaf': [best_random.min_samples_leaf - 1, best_random.min_samples_leaf,
                             best_random.min_samples_leaf + 1],
        'bootstrap': [True]
    }

    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # Fit the grid search model
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters from grid search
    best_params = grid_search.best_params_

    return best_params


# Plot the oob_error_vs_estimators curve to visualize hyperparameter tuning
def plot_oob_error_vs_estimators(X_train, y_train, min_estimators, max_estimators, max_depth):
    # Define a list of ensemble classifiers with their labels
    ensemble_clfs = [("Random Forest ({0} max_depth)".format(max_depth), RandomForestRegressor(max_depth=max_depth, oob_score=True, random_state=0))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1, 10):
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
    plt.title("OOB Error Rate for Random Forest (Max Depth: {0})".format(max_depth))
    plt.legend(loc="upper right")
    plt.grid()

    # Show the plot
    plt.show()


# Get unique years from the data
unique_years = market_agg['Year'].unique()

# Create a regression model
model = RandomForestRegressor()


# Iterate through each year
for year in unique_years:
    if year == 2020:  # Condition to limit the loop
        # Add previous year target and operation cost as additional features
        X_feature, y_target = previous_feature_adder(market_agg, year)

        # Impute based on the average for NaN values
        impute_previous_target(X_feature, market_agg)

        # Prepare the training features and target
        X_train = X_feature
        y_train = y_target

        # Create an instance of StandardScaler
        scaler = StandardScaler()

        # Fit and transform the training data using the scaler
        X_train_scaled = scaler.fit_transform(X_train)

        # Prepare the test features for prediction (test is the next year)
        X_feature_test, y_target_test = next_year_data(market_agg, year)

        # Impute based on the average for NaN values
        impute_previous_target(X_feature_test, market_agg)

        # Prepare the test features and target
        X_test = X_feature_test
        y_test = y_target_test

        # Transform the testing data using the scaler
        X_test_scaled = scaler.transform(X_test)

        # Fit the base model         ********************************************
        model.fit(X_train_scaled, y_train)

        '''# Hyperparameter tuning ********************************************
        # Perform Randomized Search
        best_random_hyperparameters = random_search(X_train_scaled, y_train, model)

        # Perform Grid Search using the best hyperparameters from Randomized Search
        best_hyperparameters = grid_search(X_train, y_train, model, best_random_hyperparameters)
        
        # Create the final best model with the hyperparameters from Grid Search
        best_model = RandomForestRegressor(
            n_estimators=best_hyperparameters['n_estimators'],
            max_depth=best_hyperparameters['max_depth'],
            min_samples_split=best_hyperparameters['min_samples_split'],
            min_samples_leaf=best_hyperparameters['min_samples_leaf'],
            bootstrap=True
        )

        # Fit the final best model on the training data
        best_model.fit(X_train_scaled, y_train)

        # Evaluate the accuracy of the best model
        best_model_accuracy = evaluate_accuracy(best_model, X_test_scaled, y_test)

        # Print the results
        print(f" Hyperparameter tuning for the year {year}:")
        print("Best Hyperparameters:", best_hyperparameters)
        print("Best Model Accuracy as combination of Randomize Search and Grid Search: {:.2f}%".format(
            best_model_accuracy))'''

        # Plot the oob_error_vs_estimators curve to visualize hyperparameter tuning
        plot_oob_error_vs_estimators(X_train, y_train, min_estimators=20, max_estimators=450, max_depth=80)

    else:
        # Return True for years 2021 and beyond
        print("The prediction for the year 2022 is under construction")

# Print the predicted target values
# print()

'''
 Hyperparameter tuning for the 2019
 Best Hyperparameters: {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 418}
Best Model Accuracy as combination of Randomize Search and Grid Search: 88.05%
 
Best Model Accuracy as combination of Randomize Search and Grid Search: 88.4516716403735
Random Search Model Accuracy: 88.44398844868977
Base Model Accuracy: 88.91881750657092
{'bootstrap': True, 'max_depth': 100, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 197}
'''

'''
 Hyperparameter tuning for the 2020
 
Best Hyperparameters: {'bootstrap': True, 'max_depth': 80, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 400}
Best Model Accuracy as combination of Randomize Search and Grid Search: 88.41%

Best Model Accuracy as combination of Randomize Search and Grid Search: 88.52721758522122
Random Search Model Accuracy: 88.57942682281488
Base Model Accuracy: 90.27660338757948
{'bootstrap': True, 'max_depth': 100, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 197}
'''
