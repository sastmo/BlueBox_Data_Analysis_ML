"""
This script focuses on local feature selection at the group-level.
At the group level, it aims to identify features that contribute to differences between clusters within the dataset.

Group-level feature selection involves treating the problem as a binary classification task.
A new 'Target' column is created, where the target cluster is assigned the value 1, while the rest of the data is
assigned 0. This transforms the problem into a binary classification scenario, enabling the identification of features
that distinguish the selected cluster from the remaining dataset.

Generally, this script provides a versatile toolkit for comprehensive data analysis, feature selection,
and binary classification using a Lasso regression-based classifier. It includes functions to load and manipulate data,
perform feature selection, and introduces classes like LassoClassifierAnalyzer for optimizing Lasso classifiers
and FeatureImportanceAnalyzer for assessing feature importance.
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# Load and data manipulate function
def load_and_manipulate_data(file_path, selected_features_v1, threshold=0.00000, cluster_number=-1):
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
    columns_to_add = ['Year', 'Program Code', 'Cluster_Labels', 'Cluster_Probabilities',
                      'TOTAL Reported and/or Calculated Marketed Tonnes']
    selected_features_v1_extend = selected_features_v1 + columns_to_add

    # Select columns based on the first version of feature selection
    selected_features_ = clustered_data.loc[:, clustered_data.columns.isin(selected_features_v1_extend)]

    # Filter out noise points or outliers based on cluster probability threshold
    selected_features_ = selected_features_[selected_features_['Cluster_Probabilities'] >= threshold]

    # Add the 'Target' column
    selected_features_['Target'] = np.where(selected_features_['Cluster_Labels'] == cluster_number, 1, 0)

    return selected_features_


# Class for analyzing and optimizing a Lasso classifier for binary classification
class LassoClassifierAnalyzer:
    """
    Class for analyzing and optimizing a Lasso classifier for binary classification tasks.

    This class provides methods to analyze and optimize a Lasso classifier for binary
    classification problems. It includes finding the best alpha (penalty parameter) for
    Lasso regularization and analyzing the Lasso classifier with the selected alpha.

    Attributes:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - X_test (array-like): Testing data features.
    - y_test (array-like): Testing data labels.

    Methods:
    - calculate_best_alpha(): Find the best alpha using cross-validation for Lasso classification.
    - analyze_lasso(alpha): Analyze the Lasso classifier with a specific alpha value.
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initialize the LassoClassifierAnalyzer class.

        Args:
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_test (array-like): Testing data features.
        - y_test (array-like): Testing data labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def calculate_best_alpha(self):
        """
        Calculate the best alpha (penalty parameter) for Lasso Classifier using cross-validation.

        Returns:
        - best_alpha (float): The best alpha value determined by cross-validation.
        """
        best_alpha = None
        best_f1 = 0  # Initialize with a low value

        alphas = np.logspace(-30, 30, 5)
        # Adjust the alpha range

        skf = StratifiedKFold(n_splits=7)  # Stratified cross-validation

        for alpha in alphas:
            lasso_classifier = LogisticRegression(penalty='l1', C=1 / alpha, solver='liblinear')

            # Perform stratified cross-validation with f1_micro scoring
            cv_scores = cross_val_score(lasso_classifier, self.X_train, self.y_train, cv=skf, scoring='f1_micro')
            mean_f1 = np.mean(cv_scores)

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_alpha = alpha

        return best_alpha

    def analyze_lasso(self, alpha):
        """
        Analyze the Lasso Classifier with a specific alpha value.

        Args:
        - alpha (float): The alpha value to use for Lasso regularization.

        Returns:
        - lasso_coef (array-like): Coefficients of the Lasso classifier.
        - lasso_intercept (float): Intercept of the Lasso classifier.
        - lasso_train_score (float): Training accuracy score of the Lasso classifier.
        - lasso_test_score (float): Testing accuracy score of the Lasso classifier.
        """
        lasso_classifier = LogisticRegression(penalty='l1', C=1 / alpha, solver='liblinear')
        lasso_classifier.fit(self.X_train, self.y_train)

        lasso_coef = lasso_classifier.coef_[0]
        lasso_intercept = lasso_classifier.intercept_[0]
        lasso_train_score = lasso_classifier.score(self.X_train, self.y_train)
        lasso_test_score = lasso_classifier.score(self.X_test, self.y_test)

        return lasso_coef, lasso_intercept, lasso_train_score, lasso_test_score


# Class for analyzing feature importance based on Lasso coefficients and visualizing the results.
class FeatureImportanceAnalyzer:
    """
    Class for analyzing feature importance based on Lasso coefficients and visualizing the results.

    This class provides methods to analyze the importance of features in a dataset
    by examining the coefficients obtained from Lasso regression and visualizing them.

    Attributes:
    - feature_names (list or None): List of feature names (or None if not available).
    - lasso_coef (array-like): Coefficients obtained from Lasso regression.

    Methods:
    - analyze_feature_importance(): Analyze and return the top features with the highest absolute Lasso coefficients.
    - plot_feature_importance(): Create a horizontal bar chart to visualize feature importance based on Lasso coefficients.
    """
    def __init__(self, feature_names, lasso_coef):
        """
        Initialize the FeatureImportanceAnalyzer class.

        Args:
        - feature_names (list or None): List of feature names (or None if not available).
        - lasso_coef (array-like): Coefficients obtained from Lasso regression.
        """
        self.feature_names = feature_names
        self.lasso_coef = lasso_coef

    def analyze_feature_importance(self):
        """
        Analyze feature importance based on Lasso coefficients.

        Returns:
        - top_features (DataFrame): DataFrame containing the top features with the highest absolute Lasso coefficients.
        """
        if self.feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(self.lasso_coef))]
        else:
            feature_names = np.array(self.feature_names)

        # Create DataFrames from the arrays
        df_feature_names = pd.DataFrame(feature_names, columns=["Feature_Name"])
        df_lasso_coef = pd.DataFrame(self.lasso_coef, columns=["Lasso_Coef"])

        # Concatenate the two DataFrames horizontally along columns
        result_df = pd.concat([df_feature_names, df_lasso_coef], axis=1)

        # Select the top 3 features with the highest absolute values for "Lasso_Coef"
        top_features = result_df.iloc[result_df['Lasso_Coef'].abs().argsort()[::-1][:3]]

        return top_features

    def plot_feature_importance(self):
        """
        Plot feature importance based on Lasso coefficients.

        This method creates a horizontal bar chart to visualize feature importance.

        Returns:
        - None
        """
        if self.feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(self.lasso_coef))]
        else:
            feature_names = self.feature_names

        # Flatten the coef array if it's multidimensional
        flat_coef = np.ravel(self.lasso_coef)

        plt.figure(figsize=(10, 5))
        plt.barh(feature_names, flat_coef)  # Use barh for horizontal bars
        plt.ylabel('Feature Name')  # Adjust the axis labels accordingly
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance')
        plt.show()


# Define the file path
file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\result_classification.csv"

# Define the selected features list
selected_features_v1 = [
    'Total Households Serviced', 'Single Family Dwellings', 'Full User Pay',
    'Bag Limit Program for Garbage Collection', 'Municipal Group', 'Single Stream',
    'Residential Promotion & Education Costs', 'Interest on Municipal Capital',
    'Total Gross Revenue', 'Interaction of Households Serviced and operation cost',
    'operation cost', 'Previous_Target', 'Program efficiency'
]

# Call the function to load and manipulate the data
selected_features = load_and_manipulate_data(file_path, selected_features_v1)

# Get unique years from the data
unique_years = selected_features['Year'].unique()

# Iterate through each year
for year in unique_years:
    if year == 2021:  # Condition to limit the loop to the year 2021

# Section 1: Prepare the dataset for analysis and standardize the features
        # Filter the dataset for the selected year (2021)
        selected_features_ = selected_features[selected_features['Year'] == year]

        # Select predictor features and the target feature
        selected_features_df = selected_features_.copy()
        X_feature = selected_features_df.drop(
            ['Target', 'Year', 'Cluster_Labels', 'Program Code', 'Cluster_Probabilities', 'Municipal Group'], axis=1
        ).copy()
        y_target = selected_features_df['Target'].copy()

        # Reset the index of X_feature and y_target before each iteration
        X_feature.reset_index(drop=True)
        y_target.reset_index(drop=True)

        # Standardize features using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_feature)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_target, test_size=0.2, random_state=0)

        # Fit a Lasso model
        # Create an instance of the LassoClassifierAnalyzer
        lasso_analyzer = LassoClassifierAnalyzer(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # Find the best alpha for Lasso Classification
        best_alpha = lasso_analyzer.calculate_best_alpha()
        print("Best alpha for Lasso Classifier:", best_alpha)

# Section 2: LassoClassifierAnalyzer for classification and determining the best alpha
        # Analyze Lasso Classifier with the best alpha
        lasso_coef, lasso_intercept, lasso_train_score, lasso_test_score = lasso_analyzer.analyze_lasso(
            alpha=best_alpha)

        # Print the results
        print("Lasso Classifier Coefficients:", lasso_coef)
        print("Lasso Classifier Intercept:", lasso_intercept)
        print("Lasso Classifier Train Score:", lasso_train_score)
        print("Lasso Classifier Test Score:", lasso_test_score)

# Section 3: Perform feature analysis for 20 times to find the most important features
        # Provide feature names if available
        feature_names = [
            'Total Households Serviced', 'Single Family Dwellings', 'Full User Pay',
            'Bag Limit Program for Garbage Collection', 'Single Stream',
            'Residential Promotion & Education Costs', 'Total Gross Revenue',
            'Interaction of Households Serviced and operation cost',
            'operation cost', 'Program efficiency',
            'TOTAL Reported and/or Calculated Marketed Tonnes', 'Previous_Target'
        ]

        num_iterations = 20

        # Initialize an empty DataFrame to store feature counts
        feature_count_df = pd.DataFrame(columns=['Feature_Name', 'Count'])

        # Repeat the process 20 times
        for _ in range(num_iterations):

            # Create an instance of FeatureImportanceAnalyzer
            feature_importance_analyzer = FeatureImportanceAnalyzer(feature_names=feature_names, lasso_coef=lasso_coef)

            # Run the method to get the top features
            top_features = feature_importance_analyzer.analyze_feature_importance()

            # Iterate over the top features DataFrame
            for index, row in top_features.iterrows():
                feature_name = row['Feature_Name']

                # Check if the feature name is in the DataFrame
                if feature_name in feature_count_df['Feature_Name'].values:
                    # If it exists, increment the count by one
                    feature_count_df.loc[feature_count_df['Feature_Name'] == feature_name, 'Count'] += 1
                else:
                    # If it doesn't exist, add it to the DataFrame with a count of one
                    feature_count_df = feature_count_df.append({'Feature_Name': feature_name, 'Count': 1},
                                                               ignore_index=True)

# Section 4: Results
        # feature_count_df contains the counts of how many times each feature appeared in the top features
        print(feature_count_df)

        # Create an instance of FeatureImportanceAnalyzer
        feature_importance_analyzer = FeatureImportanceAnalyzer(feature_names=feature_names, lasso_coef=lasso_coef)

        # Analyze feature importance
        top_features = feature_importance_analyzer.analyze_feature_importance()

        # Plot feature importance
        feature_importance_analyzer.plot_feature_importance()
