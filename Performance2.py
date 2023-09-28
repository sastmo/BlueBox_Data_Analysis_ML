# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, \
    precision_score, confusion_matrix

import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Reading the dataset

load_data = loan_data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\data.csv")

df = pd.DataFrame(load_data)
print(df.columns)
print(df.head())

# class distribution
# diagnosis: B = 0, M = 1
print(df['diagnosis'].value_counts())

# by default majority class (benign) will be negative
lb = LabelBinarizer()
df['diagnosis'] = lb.fit_transform(df['diagnosis'].values)
targets = df['diagnosis']
print(targets)

df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, targets, test_size=0.33, stratify=targets)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now X_train_scaled and X_test_scaled are the scaled features
print("Scaled X_train:", X_train_scaled)
print("Scaled X_test:", X_test_scaled)

# Distribution betwen training and test sets
print('y_train class distribution')
print(y_train.value_counts(normalize=True))

print('y_test class distribution')
print(y_test.value_counts(normalize=True))

# Optimize for sensitivity using GridSearchCV and scoring

clf = RandomForestClassifier(n_jobs=-1)

param_grid = {
    'min_samples_split': [3, 5, 10],
    'n_estimators': [100, 300],
    'max_depth': [3, 5, 15, 25],
    'max_features': [3, 5, 10, 20]
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}


def grid_search_wrapper(refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    # Creating a StratifiedKFold cross-validation object with 10 splits
    skf = StratifiedKFold(n_splits=10)

    # Creating a GridSearchCV object
    grid_search = GridSearchCV(
        clf,  # Classifier instance (not defined in the provided code snippet)
        param_grid,  # Dictionary of hyperparameters to search
        scoring=scorers,  # Dictionary of scoring metrics
        refit=refit_score,  # Scoring metric to optimize for
        cv=skf,  # Cross-validation strategy
        return_train_score=False,  # Exclude train scores in the result
        n_jobs=-1  # Number of jobs to run in parallel (-1 means using all available cores)
    )

    # Fitting the GridSearchCV object on the training data
    grid_search.fit(X_train_scaled.values, y_train.values)

    # Making predictions using the best model found by GridSearchCV
    y_pred = grid_search.predict(X_test_scaled.values)

    # Printing the best parameters found for the selected scoring metric
    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # Displaying the confusion matrix of the predictions on the test data
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

    return grid_search


# Precision-Optimized Classifier Analysis:
grid_search_clf = grid_search_wrapper(refit_score='precision_score')

results = pd.DataFrame(grid_search_clf.cv_results_)
results = results.sort_values(by='mean_test_precision_score', ascending=False)
results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score',
         'param_max_depth', 'param_max_features', 'param_min_samples_split',
         'param_n_estimators']].head()

# Recall-Optimized Classifier Analysis
grid_search_clf = grid_search_wrapper(refit_score='recall_score')

results = pd.DataFrame(grid_search_clf.cv_results_)
results = results.sort_values(by='mean_test_recall_score', ascending=False)
results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score',
         'param_max_depth', 'param_max_features', 'param_min_samples_split',
         'param_n_estimators']].head()

# The goal in cancer diagnosis is typically to minimize false negatives
# (positives classified as negatives) as much as possible. Therefore, a classifier optimized for recall is preferred,
# as it aims to minimize false negatives even if it results in a higher number of false positives.


# Get the probability scores for class 1 from the classifier
y_scores = grid_search_clf.predict_proba(X_test_scaled)[:, 1]


# Define a function to adjust class predictions based on threshold
def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


# Generate the precision-recall curve
p, r, thresholds = precision_recall_curve(y_test, y_scores)


# Define a function to visualize the precision-recall curve and the current threshold
def precision_recall_threshold(t=0.5):
    """
    Plots the precision-recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    # Generate new class predictions based on the adjusted_classes function
    y_pred_adj = adjusted_classes(y_scores, t)

    # Display the confusion matrix for the adjusted predictions
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'],
                       index=['neg', 'pos']))

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2, where='post')
    plt.fill_between(r, p, step='post', alpha=0.2, color='b')
    plt.ylim([0.5, 1.01])
    plt.xlim([0.5, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Plot the current threshold on the curve
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k', markersize=15)


# Call the precision_recall_threshold function to fine-tune the threshold
precision_recall_threshold(0.17)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Plots precision and recall scores as a function of the decision threshold.
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores vs. Decision Threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


# Use the same precision (p), recall (r), and thresholds calculated earlier
plot_precision_recall_vs_threshold(p, r, thresholds)

# Calculate the False Positive Rate (fpr), True Positive Rate (tpr), and AUC thresholds
fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores)

# Print the AUC of ROC
print("AUC of ROC:", auc(fpr, tpr))


# Plot the ROC curve
def plot_roc_curve(fpr, tpr, label=None):
    """
    Plots the ROC curve.
    """
    plt.figure(figsize=(8, 8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')


plot_roc_curve(fpr, tpr, label='recall_optimized')
