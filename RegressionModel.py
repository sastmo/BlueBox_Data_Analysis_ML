import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
# from Data_Model import market_agg
# from Hyperparameter_tune import best_params


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


# Best Hyperparameters resulted form Hyperparameter tuning
best_params = {'bootstrap': True, 'max_depth': 80, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 400}

# Create a regression model
model = RandomForestRegressor(**best_params)

# Lists to store evaluation metrics
r2ts_list = []
r2vs_list = []
mse_list = []
rmse_list = []
mae_list = []
r2_list = []

# Get unique years from the data
unique_years = market_agg['Year'].unique()

# Iterate through each year
for year in unique_years:
    if year < 2021:  # Condition to limit the loop
        # Add previous year target and operation cost as additional features
        X_feature, y_target = previous_feature_adder(market_agg, year)

        # Impute based on the average for NaN values
        impute_previous_target(X_feature, market_agg)

        # Prepare the training features and target
        X_train = X_feature
        y_train = y_target

        X_train = pd.get_dummies(X_train, columns=['Municipal Group'], prefix='Municipal Group')

        # Create an instance of StandardScaler
        scaler = StandardScaler()

        # Fit and transform the training data using the scaler
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit the model
        model.fit(X_train_scaled, y_train)

        # Prepare the test features for prediction (test is the next year)
        X_feature_test, y_target_test = next_year_data(market_agg, year)

        # Impute based on the average for NaN values
        impute_previous_target(X_feature_test, market_agg)

        # Prepare the test features and target
        X_test = X_feature_test
        y_test = y_target_test

        X_test = pd.get_dummies(X_test, columns=['Municipal Group'], prefix='Municipal Group')

        # Transform the testing data using the scaler
        X_test_scaled = scaler.transform(X_test)

        # Predict on the test data
        predicted_target = model.predict(X_test_scaled)

        # Combine X_test, y_test, and predicted_target into a DataFrame
        result = X_feature_test[['Program Code', 'Municipal Group']]
        y_test_values = y_test.values
        result['y_target'] = y_test_values
        result['predicted_target'] = predicted_target

        # Display the Prediction Result
        print(f"Prediction for year {year + 1}:")
        print(result)

        # Update the data with predicted Target values for the specific year
        market_agg.loc[market_agg['Year'] == year + 1, 'Predicted_Target'] = predicted_target[0]

        # Get feature importance scores
        feature_importance = model.feature_importances_
        print(f"Year: {year}")
        for feature_name, importance_score in zip(X_train.columns, feature_importance):
            print(f"  Feature: {feature_name}, Importance: {importance_score:.4f}")

        # Plot feature importance
        plt.figure(figsize=(8, 6))
        plt.bar(X_train.columns, feature_importance)
        plt.title(f"Feature Importance for Year {year}")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.show()

        # Calculate evaluation metrics
        # Print OOB Score
        print(f"OOB Score: {model.oob_score:.2f}")

        r2ts = model.score(X_train_scaled, y_train)
        r2vs = model.score(X_test_scaled, y_test)
        mse = mean_squared_error(y_test, predicted_target)
        rmse = mean_squared_error(y_test, predicted_target, squared=False)
        mae = mean_absolute_error(y_test, predicted_target)
        r2 = r2_score(y_test, predicted_target)

        # Append metrics to lists
        r2ts_list.append(r2ts)
        r2vs_list.append(r2vs)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    else:
        # Return True for years 2021 and beyond
        print("The prediction for the year 2022 is under construction")

# Print the evaluation metrics for each year
for year, r2ts, r2vs, mse, rmse, mae, r2 in zip(unique_years, r2ts_list, r2vs_list, mse_list, rmse_list, mae_list, r2_list):
    print(f"R^2 Training Score: {r2ts:.2f}")
    print(f"R^2 Validation Score: {r2vs:.2f}")
    print(f"Evaluation metrics for year {year + 1}:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")
    print("-" * 40)

# Print the predicted target values
# print(market_agg)
