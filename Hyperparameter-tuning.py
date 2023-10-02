import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


# Calculate Variance Inflation Factor (VIF)
def calculate_vif(data_frame):
    """
    Calculate the Variance Inflation Factor (VIF) for each variable in the DataFrame.

    Parameters:
    - data_frame (DataFrame): The DataFrame containing numerical variables.

    Returns:
    - vif_data_ (DataFrame): DataFrame with columns 'Variable' and 'VIF' representing VIF values for each variable.
    """
    vif_data_ = pd.DataFrame()
    vif_data_["Variable"] = data_frame.columns
    vif_data_["VIF"] = [variance_inflation_factor(data_frame.values, i) for i in range(data_frame.shape[1])]
    return vif_data_


# Random Search for Initial Hyperparameter Tuning
def random_search(X_train, y_train, model):
    """
    Perform Randomized Search to find the initial range of hyperparameters for a regression model.

    Parameters:
    - X_train: The training features.
    - y_train: The training labels or target values.
    - model: The base regression model.

    Returns:
    - best_random: The best regression model with hyperparameters from Randomized Search.
    """
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


# Grid Search for Fine-Tuning Hyperparameters
def grid_search(X_train, y_train, model, best_random):
    """
    Perform Grid Search to fine-tune hyperparameters for a regression model based on initial parameters from Randomized Search.

    Parameters:
    - X_train: The training features.
    - y_train: The training labels or target values.
    - model: The base regression model.
    - best_random: The best regression model from Randomized Search with initial hyperparameters.

    Returns:
    - best_params: The best hyperparameters found through Grid Search.
    """
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


# Evaluate Hyperparameter Accuracy
def evaluate_accuracy(model, features, labels):
    """
    Evaluate the accuracy of a regression model using the Mean Absolute Error (MAE).

    Parameters:
    - model: The regression model to be evaluated.
    - features: The input features for prediction.
    - labels: The true labels or target values.

    Returns:
    - accuracy: The accuracy of the model in percentage.
    """
    predictions = model.predict(features)
    mae = mean_absolute_error(labels, predictions)
    accuracy = 100 * (1 - mae / np.mean(labels))
    return accuracy


# Plot OOB Error vs. Number of Estimators Curve
def plot_oob_error_vs_estimators(X_train, y_train, min_estimators, max_estimators, max_depth):
    """
    Generate a plot of Out-of-Bag (OOB) Error Rate vs. Number of Estimators to visualize hyperparameter tuning.

    Parameters:
    - X_train: The training features.
    - y_train: The training labels or target values.
    - min_estimators: The minimum number of estimators (trees) to consider.
    - max_estimators: The maximum number of estimators (trees) to consider.
    - max_depth: The maximum depth of the Random Forest model.

    This function plots the OOB Error Rate vs. the number of estimators (trees) to help visualize the impact of
    hyperparameter tuning on the Random Forest model's performance.
    """
    # Define a list of ensemble classifiers with their labels
    ensemble_clfs = [("Random Forest ({0} max_depth)".format(max_depth),
                      RandomForestRegressor(max_depth=max_depth, oob_score=True, random_state=0))]

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


# Load data:
# Define the file path
file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\result_classification.csv"

# Read the CSV file into a DataFrame
recycle_material = pd.read_csv(file_path)

# Get unique years from the data
unique_years = recycle_material['Year'].unique()

# Iterate through each year to improve the model, handle multicollinearity, prepare training and test data,
# perform hyperparameter tuning using Randomized Search and Grid Search, and visualize OOB Error vs Estimators curve.
# Iterate through each year
for year in unique_years:
    if year < 2021:  # Condition to limit the loop

# Section 1: Model Improvement and Handling Multicollinearity👇
        # Remove irrelevant columns and handle multicollinearity
        recycle_material_filtered = recycle_material.drop(columns=['Year', 'Cluster_Probabilities', 'Cluster_Labels',
                                                                   'TOTAL Reported and/or Calculated Marketed Tonnes',
                                                                   'Quadrant', 'Program Code'])

        # Handle missing values and calculate VIF
        numerical_data = recycle_material_filtered.select_dtypes(include=[np.number])
        numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)
        numerical_data.dropna(inplace=True)
        vif_data = calculate_vif(numerical_data)
        print(vif_data)

# Section 2: Prepare Test and Training Data for Regression using Random Forest👇

        # Add previous year target and operation cost as additional features
        X_feature, y_target = feature_target_selection(recycle_material, year)

        # Reset the index of X before each iteration
        X_feature.reset_index(drop=True)

        # Reset the index of y_target before each iteration
        y_target.reset_index(drop=True)

        # Prepare the training features and target
        X_train = X_feature.drop(['Year', 'Program Code', 'Cluster_Probabilities', 'Cluster_Labels',
                                  'Quadrant'], axis=1).copy()
        y_train = y_target

        # Create an instance of StandardScaler
        scaler = StandardScaler()

        # Fit and transform the training data using the scaler
        X_train_scaled = scaler.fit_transform(X_train)

        # Prepare the test features for prediction (test is the next year)
        next_year = year + 1
        X_feature_test, y_target_test = feature_target_selection(recycle_material, next_year)

        # Reset the index of X before each iteration
        X_feature_test.reset_index(drop=True)

        # Reset the index of y_target before each iteration
        y_target_test.reset_index(drop=True)

        # Prepare the test features and target
        X_test = X_feature_test.drop(['Year', 'Program Code', 'Cluster_Probabilities', 'Cluster_Labels',
                                      'Quadrant'], axis=1).copy()
        y_test = y_target_test

        # Transform the testing data using the scaler
        X_test_scaled = scaler.transform(X_test)

        # Fit the base model    ****************************************************************
        # Create a regression model
        model = RandomForestRegressor()
        model.fit(X_train_scaled, y_train)

# Section 3: Hyperparameter Tuning using Randomized Search and Grid Search 👇
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

        # Fit the final best model on the training data ********************************************
        best_model.fit(X_train_scaled, y_train)

        # Evaluate the accuracy of the best model
        best_model_accuracy = evaluate_accuracy(best_model, X_test_scaled, y_test)

        # Print the results
        print(f" Hyperparameter tuning for the year {year}:")
        print("Best Hyperparameters:", best_hyperparameters)
        print("Best Model Accuracy as combination of Randomize Search and Grid Search: {:.2f}%".format(
            best_model_accuracy))

# Section 4: Perform OOB Error vs Estimators Curve Visualization and Hyperparameter Tuning👇
        # Plot the oob_error_vs_estimators curve to visualize hyperparameter tuning
        plot_oob_error_vs_estimators(X_train, y_train, min_estimators=20, max_estimators=450, max_depth=80)

    else:
        # Return True for years 2021 and beyond
        print("The prediction for the year 2022 is under construction")



