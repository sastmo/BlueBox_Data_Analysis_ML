# https://chat.openai.com/share/271c4309-05aa-40a1-90c8-fdd8bea47972
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut
import matplotlib.pyplot as plt


def holdout_method(X_train, X_test, y_train, y_test):
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return predictions, mse, r2


def k_folds_cross_validation(df, y, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(df):
        X_train, X_cv = df.iloc[train_index], df.iloc[test_index]
        y_train, y_cv = y[train_index], y[test_index]

        lm = linear_model.LinearRegression()
        model = lm.fit(X_train, y_train)
        predictions_cv = lm.predict(X_cv)

        mse_cv = mean_squared_error(y_cv, predictions_cv)
        r2_cv = r2_score(y_cv, predictions_cv)

        mse_scores.append(mse_cv)
        r2_scores.append(r2_cv)

    return np.mean(mse_scores), np.mean(r2_scores)  # Return means, not predictions_cv


def k_folds_optimization(df, y, max_k=11):
    mse_scores = []
    r2_scores = []

    for k in range(2, max_k):
        mse, r2 = k_folds_cross_validation(df, y, k)  # Call the correct function
        mse_scores.append(mse)
        r2_scores.append(r2)

    return mse_scores, r2_scores


def loocv(df, y):
    loo = LeaveOneOut()
    mse_scores = []
    r2_scores = []

    for train_index, test_index in loo.split(df):
        X_train, X_cv = df.iloc[train_index], df.iloc[test_index]
        y_train, y_cv = y[train_index], y[test_index]

        lm = linear_model.LinearRegression()
        model = lm.fit(X_train, y_train)
        predictions_cv = lm.predict(X_cv)

        mse_cv = mean_squared_error(y_cv, predictions_cv)
        r2_cv = r2_score(y_cv, predictions_cv)

        mse_scores.append(mse_cv)
        r2_scores.append(r2_cv)

    return np.mean(mse_scores), np.mean(r2_scores)


# Load the Diabetes dataset
columns = "age sex bmi map tc ldl hdl tch ltg glu".split()
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# Holdout Method
holdout_predictions, holdout_mse, holdout_r2 = holdout_method(X_train, X_test, y_train, y_test)
print("Holdout Method MSE:", holdout_mse)
print("Holdout Method R^2 Score:", holdout_r2)

# K-Folds Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse, cv_r2 = k_folds_cross_validation(df, y, k=5)  # Only get means, not predictions
print("K-Folds Cross Validation Mean MSE:", cv_mse)
print("K-Folds Cross Validation Mean R^2 Score:", cv_r2)

# Create plots comparing actual vs. prediction for both methods
plt.figure(figsize=(12, 6))

# Holdout Method Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, holdout_predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Holdout Method: Actual vs. Predicted")

# K-Folds Cross Validation Plot
plt.subplot(1, 2, 2)
for train_index, test_index in kf.split(df):
    X_train, X_cv = df.iloc[train_index], df.iloc[test_index]
    y_train, y_cv = y[train_index], y[test_index]

    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions_cv = lm.predict(X_cv)

    plt.scatter(y_cv, predictions_cv)  # Use y_cv here
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("K-Folds CV: Actual vs. Predicted")

plt.tight_layout()
plt.show()

# Perform K-Folds Cross Validation for different K values
max_k = 11  # Set the maximum K value
mse_scores, r2_scores = k_folds_optimization(df, y, max_k)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(2, max_k), mse_scores, marker='o')
plt.xlabel("Number of Folds (K)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("K-Folds Cross Validation: MSE vs. K")

plt.subplot(1, 2, 2)
plt.plot(range(2, max_k), r2_scores, marker='o', color='r')
plt.xlabel("Number of Folds (K)")
plt.ylabel("R^2 Score")
plt.title("K-Folds Cross Validation: R^2 Score vs. K")

plt.tight_layout()
plt.show()

# Calculate LOOCV MSE and R²
loocv_mse, loocv_r2 = loocv(df, y)
print("LOOCV Mean MSE:", loocv_mse)
print("LOOCV Mean R^2 Score:", loocv_r2)
