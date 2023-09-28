import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


def load_and_manipulate_data(file_path, selected_features_v1, threshold=0.00000, cluster_number=5):
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
                      'Program efficiency', 'TOTAL Reported and/or Calculated Marketed Tonnes']
    selected_features_v1_extend = selected_features_v1 + columns_to_add

    # Select columns based on the first version of feature selection
    selected_features_ = clustered_data.loc[:, clustered_data.columns.isin(selected_features_v1_extend)]

    # Filter out noise points or outliers based on cluster probability threshold
    selected_features_ = selected_features_[selected_features_['Cluster_Probabilities'] >= threshold]

    # Add the 'Target' column
    selected_features_['Target'] = np.where(selected_features_['Cluster_Labels'] == cluster_number, 1, 0)

    return selected_features_


def extract_alpha_and_plot(X_train, X_test, y_train, y_test):
    # Extract different values of alpha for pruning
    clf_dt = DecisionTreeClassifier(random_state=0)
    clf_dt = clf_dt.fit(X_train, y_train)
    path = clf_dt.cost_complexity_pruning_path(X_train, y_train)

    # Extract different values for alpha and sort them in ascending order
    ccp_alphas = np.sort(path.ccp_alphas)
    ccp_alphas = ccp_alphas[:-1]  # exclude the maximum value for alpha

    print("-_-" * 25)
    print(ccp_alphas)
    print("-_-" * 25)

    clf_dts = []  # Create an array to store decision trees

    # Create decision trees for different values of alpha
    for ccp_alpha in ccp_alphas:
        clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf_dt.fit(X_train, y_train)
        clf_dts.append(clf_dt)

    # Graph the accuracy of trees using the Training and Testing Datasets as a function of alpha
    train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
    test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()

    print(f"first alpha for pruning: { 0.01702452:.5f}")

    return ccp_alphas,  0.01702452


def cross_validate_alpha(X_train, y_train, alpha_values):
    """
    Perform cross-validation to find the best alpha value for cost complexity pruning.

    Parameters:
        X_train (array-like): The training data features.
        y_train (array-like): The training data labels.
        alpha_values (list): List of alpha values to evaluate.

    Returns:
        best_alpha (float): The best alpha value found through cross-validation.
    """
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha_values)  # create the tree with ccp_alpha=0.097

    # Using 5-fold cross validation create 5 different training and testing datasets that
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})

    df.plot(x='tree', y='accuracy', marker='o', linestyle='--')
    plt.show()


def find_ideal_ccp_alpha(ccp_alphas, X_train, y_train):
    """
    Find the ideal ccp_alpha using cross-validation.

    Parameters:
        ccp_alphas (list): List of candidate alpha values.
        X_train (array-like): The training data features.
        y_train (array-like): The training data labels.

    Returns:
        ideal_ccp_alpha (float): The ideal ccp_alpha value.
    """
    alpha_loop_values = []

    for ccp_alpha in ccp_alphas:
        clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
        alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

    alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])

    # Plot the mean accuracy with error bars
    alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
    plt.show()

    # Find the range of alpha values that produce high mean accuracy
    ideal_range = alpha_results[(alpha_results['alpha'] > 0.00000) & (alpha_results['alpha'] < 0.02686203)]

    # Get the ideal alpha value from the range
    ideal_ccp_alpha = alpha_results.sort_values(by='mean_accuracy', ascending=False)['alpha'].values[0]
    print(f"ideal alpha for pruning: {ideal_ccp_alpha:.5f}")

    return ideal_ccp_alpha


def evaluate_and_visualize_tree(X_feature, X_train, X_test, y_train, y_test, ideal_alpha):
    """
    Evaluate and visualize a pruned decision tree.

    Parameters:
    - X_train: Training feature data.
    - X_test: Testing feature data.
    - y_train: Training target labels.
    - y_test: Testing target labels.
    - ideal_alpha: The optimal alpha value for pruning.

    This function builds and trains a pruned decision tree using the specified alpha
    and minimum samples per leaf, displays the confusion matrix with custom labels,
    calculates and prints additional performance metrics, and visualizes the pruned
    decision tree.

    Returns:
    None
    """
    # Build and train a new decision tree with the specified alpha and minimum samples per leaf
    clf_dt_pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=ideal_alpha)
    clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

    # Display the confusion matrix with custom labels
    cm_display = ConfusionMatrixDisplay.from_estimator(
        clf_dt_pruned, X_test, y_test, display_labels=["Not in Cluster", "In Cluster"]
    )
    plt.show()

    # Calculate and print additional performance metrics
    tn, fp, fn, tp = cm_display.confusion_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.2f}")
    print(f"Specificity (True Negative Rate): {specificity:.2f}")

    # Get the feature names as a list
    feature_names = X_feature.columns.tolist()

    # Set the figure size
    plt.figure(figsize=(15, 7.5))

    # Plot the pruned decision tree
    plot_tree(
        clf_dt_pruned,
        filled=True,
        rounded=True,
        class_names=["Not in Cluster", "In Cluster"],
        feature_names=feature_names
    )

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

# Print selected features columns and count
print(selected_features.columns)
print(selected_features.count())
print(selected_features['Cluster_Probabilities'].mean())

# Get unique years from the data
unique_years = selected_features['Year'].unique()

# Iterate through each year
for year in unique_years:
    if year == 2021:  # Condition to limit the loop to the year 2021

        # For the initial year, use the mean of the target from the same Program Code
        selected_features_ = selected_features[selected_features['Year'] == year]

        # Select predictor features and target feature
        selected_features_df = selected_features_.copy()

        X_feature = selected_features_df.drop(
            ['Target', 'Year', 'Cluster_Labels', 'Program Code', 'Cluster_Probabilities'], axis=1).copy()

        X_feature = pd.get_dummies(X_feature, columns=['Municipal Group'], prefix='Municipal Group')

        y_target = selected_features_df['Target'].copy()

        # Reset the index of X before each iteration
        X_feature.reset_index(drop=True)

        # Reset the index of y_target before each iteration
        y_target.reset_index(drop=True)

        # Standardize features using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_feature)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_target, test_size=0.2,
                                                            random_state=0)

        # Extract alpha values and plot the results
        ccp_alphas, alpha1 = extract_alpha_and_plot(X_train, X_test, y_train, y_test)
        print(ccp_alphas)

        # Cross-validate alpha values
        cross_validate_alphas = cross_validate_alpha(X_train, y_train, alpha1)

        # Find the ideal alpha value
        ideal_alpha = find_ideal_ccp_alpha(ccp_alphas, X_train, y_train)

        # Build and train a new decision tree, utilizing the optimal alpha value
        evaluate_and_visualize_tree(X_feature, X_train, X_test, y_train, y_test, ideal_alpha)
