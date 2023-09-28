import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from Feature_Selection import previous_feature_adder, impute_previous_target
# from Data_Model import market_agg
# from Feature_Selection2 import selected_features_df


# Load data :
# Define the file path
file_path_market_agg = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\market_agg.csv"

# Read the CSV file into a DataFrame
market_agg = pd.read_csv(file_path_market_agg)

selected_features_v1 = np.array(['Total Households Serviced', 'Single Family Dwellings', 'Full User Pay',
                                 'Bag Limit Program for Garbage Collection', 'Municipal Group', 'Single Stream',
                                 'Residential Promotion & Education Costs', 'Interest on Municipal  Capital',
                                 'Total Gross Revenue', 'Interaction of Households Serviced and operation cost',
                                 'Program efficiency', 'operation cost', 'Previous_Target']
                                )
# Columns to add
columns_to_add = np.array(['Year', 'Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes'])

# Append the new columns to selected_features_v1
selected_features_v1_extend = np.append(selected_features_v1, columns_to_add)

# Use loc to select columns based on the first version of feature selection
selected_features_df = market_agg.loc[:, market_agg.columns.isin(selected_features_v1_extend)]


# Function to evaluate feature importance and select features
def evaluate_feature_importance(X_feature, y_target, model):
    X_feature = X_feature.copy()

    # Impute NaN values based on average
    X_feature = impute_previous_target(X_feature, market_agg)

    # Drop unwanted columns
    X_feature = X_feature.drop(columns=['Year', 'Program Code'])

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_feature)

    # Fit Lasso model
    model.fit(X_train_scaled, y_target)

    print("Best alpha using built-in LassoCV: %f" % model.alpha_)
    print("Best score using built-in LassoCV: %f" % model.score(X_train_scaled, y_target))

    coef = pd.Series(model.coef_, index=X_feature.columns)
    selected_features = X_feature.columns[coef != 0]

    print("Selected Features:", selected_features)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")

    imp_coef = coef.sort_values()
    plt.figure(figsize=(8.0, 10.0))
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Lasso Model")
    plt.show()


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

        # Create a regression model
        model = LassoCV()

        # Call the function to evaluate feature importance and select features
        evaluate_feature_importance(X_feature, y_target, model)

        # Store selected features in the dictionary
        selected_features_dict[year] = X_feature.columns
    else:
        print("The new Version is under construction")
