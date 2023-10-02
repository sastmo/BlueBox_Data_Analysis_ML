import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from treeinterpreter import treeinterpreter as ti, utils
import lime.lime_tabular


def load_and_manipulate_data(file_path, selected_features_v1):
    """
    Load data from a CSV file, manipulate it, and return the selected features.

    Args:
    - file_path (str): Path to the CSV file.
    - selected_features_v1 (list): List of feature names to select.
    - threshold (float): Threshold to filter out noise points based on cluster probability.
    - cluster_number (int): The cluster number to identify for the 'Target' column.

    Returns:
    - selected_features (DataFrame): DataFrame containing the selected features.
    """

    # Read the CSV file into a DataFrame
    clustered_data = pd.read_csv(file_path)

    # Append the new columns to selected_features_v1
    columns_to_add = ['Year', 'Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes']
    selected_features_v1_extend = selected_features_v1 + columns_to_add

    # Select columns based on the first version of feature selection
    selected_features_ = clustered_data.loc[:, clustered_data.columns.isin(selected_features_v1_extend)]

    return selected_features_


# 1. Observation Level Feature Importance using Treeinterpreter
def analyze_feature_contributions_treeinterpreter(model, X_train_scaled2, X_feature, program_codes_for_compression):
    """
    Analyze feature contributions for a machine learning model using TreeInterpreter.

    Args:
        model (object): The trained machine learning model.
        X_train_scaled2 (DataFrame): The scaled training data.
        X_feature (DataFrame): The feature data.
        program_codes_for_compression (list): List of program codes for analysis.
    """
    # Step 1: Using TreeInterpreter to calculate feature contributions for each row
    prediction, bias, contributions = ti.predict(model, X_train_scaled2)

    # Step 2: Interpreting Feature Contributions for each row
    for i in range(len(program_codes_for_compression)):
        print("Row", program_codes_for_compression[i])
        print("Bias (trainset mean)", bias[i])
        print("Feature contributions:")
        for c, feature in sorted(zip(contributions[i], X_feature.columns), key=lambda x: -abs(x[0])):
            print(feature, round(c, 2))
        print("-" * 20)

    # Step 3: Aggregated Contribution Analysis between two rows
    prediction1, bias1, contributions1 = ti.predict(model, np.array([X_feature.iloc[0]]), joint_contribution=True)
    prediction2, bias2, contributions2 = ti.predict(model, np.array([X_feature.iloc[1]]), joint_contribution=True)
    aggregated_contributions1 = utils.aggregated_contribution(contributions1)
    aggregated_contributions2 = utils.aggregated_contribution(contributions2)
    res = []
    for k in set(aggregated_contributions1.keys()).union(set(aggregated_contributions2.keys())):
        res.append(([X_feature.columns[k] for k in k],
                    aggregated_contributions1.get(k, 0) - aggregated_contributions2.get(k, 0)))
    for lst, v in (sorted(res, key=lambda x: -abs(x[1])))[:10]:
        print(lst, v)


# 2.Observation Level Feature Importance using LIME (Local Interpretable Model-agnostic Explanations)
def analyze_feature_contributions_LIME(model, X_train_, X_feature, program_codes_for_compression):

    X_train = pd.DataFrame(X_train_)
    X_train.columns = X_feature.columns
    # Create a LimeTabularExplainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                       mode='regression',
                                                       feature_names=X_train.columns,
                                                       categorical_features=[],
                                                       categorical_names=[],
                                                       discretize_continuous=True)

    exp_program_codes_for_compression = [None] * len(program_codes_for_compression)
    explanation_text_program_codes_for_compression = [None] * len(program_codes_for_compression)
    explanation_plot_program_codes_for_compression = [None] * len(program_codes_for_compression)

    for i_, program_code in enumerate(program_codes_for_compression):
        # Explain the observation for the given program code
        exp_program_codes_for_compression[i_] = explainer.explain_instance(
            X_train.values[i_],
            model.predict,
            num_features=5
        )

        # Get the explanation as a list of tuples
        explanation_text_program_codes_for_compression[i_] = exp_program_codes_for_compression[i_].as_list()

        # Get the explanation as a pyplot figure
        explanation_plot_program_codes_for_compression[i_] = exp_program_codes_for_compression[i_].as_pyplot_figure()

        print(f"Explanation for Observation the program {program_code}:")
        print(explanation_text_program_codes_for_compression[i_])
        plt.title(f"Feature observation for the program {program_code}")
        plt.show()


# Define the file path
file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\market_agg.csv"

# Define the selected features list
selected_features_v1 = [
    'Total Households Serviced', 'Single Family Dwellings', 'Full User Pay',
    'Bag Limit Program for Garbage Collection', 'Municipal Group', 'Single Stream',
    'Residential Promotion & Education Costs', 'Interest on Municipal Capital',
    'Total Gross Revenue', 'Interaction of Households Serviced and operation cost',
    'operation cost', 'Previous_Target',  'Program efficiency'
]

# Call the function to load and manipulate the data
selected_features = load_and_manipulate_data(file_path, selected_features_v1)

# Best Hyperparameters resulted form Hyperparameter tuning
best_params = {'bootstrap': True, 'max_depth': 90, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 489}

# Create a regression model
model = RandomForestRegressor(**best_params)

# Get unique years from the data
unique_years = selected_features['Year'].unique()

# Iterate through each year
for year in unique_years:
    if year == 2021:
        # For the initial year, use the mean of the target from the same Program Code
        selected_features_ = selected_features[selected_features['Year'] == year]

        # Select predictor features and target feature
        selected_features_df = selected_features_.copy()

        # Selected relevant features and target for use in the model
        y_target_global = selected_features_df['TOTAL Reported and/or Calculated Marketed Tonnes'].copy()
        X_feature_global = selected_features_df.drop(
            ['Year', 'Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes'], axis=1).copy()

        X_feature_global_temp = selected_features_df.drop(
            ['Year', 'TOTAL Reported and/or Calculated Marketed Tonnes'], axis=1).copy()

        X_feature_global = pd.get_dummies(X_feature_global, columns=['Municipal Group'], prefix='Municipal Group')
        X_feature_global_temp = pd.get_dummies(X_feature_global_temp, columns=['Municipal Group'], prefix='Municipal Group')

        # Standardize features using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_feature_global)

        # Regression Model Using RandomForest
        model.fit(X_train_scaled, y_target_global)

        # Select Programs for compression based on Data Exploratory Analysis
        program_code1 = 441
        program_code2 = 36
        program_codes_for_compression = [program_code1, program_code2]

        # Selected rows for analysis
        X_feature = X_feature_global_temp[X_feature_global_temp['Program Code'].isin(program_codes_for_compression)]
        X_feature = X_feature.drop(['Program Code'], axis=1).copy()

        y_target = selected_features_df[selected_features_df['Program Code'].isin(program_codes_for_compression)][
            'TOTAL Reported and/or Calculated Marketed Tonnes'].copy()

        # Standardize features using StandardScaler
        X_train_scaled2 = scaler.fit_transform(X_feature)

        # Analyze local feature Contribution using Treeinterpreter
        analyze_feature_contributions_treeinterpreter(model, X_train_scaled2, X_feature, program_codes_for_compression)

        # Analyze local feature Contribution using LIME
        analyze_feature_contributions_LIME(model, X_train_scaled2, X_feature, program_codes_for_compression)
