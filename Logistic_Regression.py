# https://chat.openai.com/share/c81d6e47-5c64-4d7f-9d4f-7918adadebd3
# https://realpython.com/logistic-regression-python/
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# https://www.statsmodels.org/stable/discretemod.html

import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class LogisticRegressionAnalysis:
    def __init__(self, dataset):
        self.pseudo_r_squared = None
        self.dataset = dataset.copy()
        self.preprocess_data()
        self.perform_preliminary_analysis()
        self.split_data()
        self.perform_logistic_regression()
        self.print_summary()
        self.calculate_pseudo_r_squared()
        self.plot_sigmoid_curve_with_data()
        self.plot_sigmoid_curve_only()
        self.calculate_accuracy()
        self.print_evaluation_metrics()

    def preprocess_data(self):
        # Remove rows with missing values in important columns
        self.dataset = self.dataset.dropna(subset=["sex", "age", "pclass", "fare", "embarked"])
        self.dataset["sex"] = self.dataset["sex"].map({"female": "F", "male": "M"})
        self.dataset["sex"] = pd.Categorical(self.dataset["sex"])
        self.dataset["pclass"] = pd.Categorical(self.dataset["pclass"])

    def perform_preliminary_analysis(self):
        cross_tab = pd.crosstab(self.dataset["sex"], self.dataset["survived"])
        odds_data = cross_tab.copy()
        odds_data["Odds"] = (cross_tab[1] / cross_tab[0]).round(2)
        combined_odds_ratio = (odds_data.loc["F", "Odds"] / odds_data.loc["M", "Odds"]).round(2)

        print("Preliminary Analysis of Odds Ratios:")
        print(odds_data)
        print("\nCombined Odds Ratio:", combined_odds_ratio)

    def split_data(self):
        X = self.dataset[["sex", "pclass", "age", "fare", "embarked"]]
        self.X_encoded = pd.get_dummies(X, columns=["sex", "pclass", "embarked"], drop_first=True)
        self.y = self.dataset["survived"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_encoded, self.y, test_size=0.3, random_state=23
        )

        self.X_test_male = self.X_test[self.X_test['sex_M'] == 1]
        self.X_test_female = self.X_test[self.X_test['sex_M'] == 0]
        self.y_test_male = self.y_test[self.X_test['sex_M'] == 1]
        self.y_test_female = self.y_test[self.X_test['sex_M'] == 0]

    def perform_logistic_regression(self):
        X_train_const = sm.add_constant(self.X_train)
        logit_model = sm.Logit(self.y_train, X_train_const)
        self.logit_result = logit_model.fit()
        X_test_const = sm.add_constant(self.X_test)
        self.y_prob = self.logit_result.predict(X_test_const)

    def print_summary(self):
        print("\nLogistic Regression Summary:")
        print(self.logit_result.summary())

    def calculate_pseudo_r_squared(self):
        ll_null = self.logit_result.llnull
        ll_proposed = self.logit_result.llf
        self.pseudo_r_squared = 1 - (ll_proposed / ll_null)
        print("\nMcFadden's Pseudo R-squared:", self.pseudo_r_squared)

    def plot_sigmoid_curve_with_data(self):
        # Define the sigmoid function
        def sigmoid(x, k, x0):
            return 1 / (1 + np.exp(-k * (x - x0)))

        # Initial parameter values
        init_params = [1, 1]

        # Fit sigmoid curve using curve_fit
        params, _ = curve_fit(sigmoid, self.X_test.iloc[:, 1], self.y_test, p0=init_params)

        # Generate the fitted curve using estimated parameters
        x_range = np.linspace(min(self.X_test.iloc[:, 1]), max(self.X_test.iloc[:, 1]), 500)
        y_out = sigmoid(x_range, params[0], params[1])

        # Plot the actual data and the fitted sigmoid curve
        plt.figure(figsize=(10, 6))

        # Plot the actual data points for males and females with different colors
        plt.scatter(self.X_test_male.iloc[:, 1], self.y_test_male, marker='o', alpha=0.5, label='Male Data',
                    color='blue')
        plt.scatter(self.X_test_female.iloc[:, 1], self.y_test_female, marker='x', alpha=0.5, label='Female Data',
                    color='red')

        # Plot the fitted sigmoid curve
        plt.plot(x_range, y_out, label='Fitted Sigmoid Curve', color='black')

        plt.xlabel('Predictor')
        plt.ylabel('Predicted Probability')
        plt.title('Fitted Sigmoid Curve with Male and Female Data')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_sigmoid_curve_only(self):
        # Define the sigmoid function
        def sigmoid(x, k, x0):
            return 1 / (1 + np.exp(-k * (x - x0)))

        # Initial parameter values
        init_params = [1, 1]

        # Fit sigmoid curve using curve_fit
        params, _ = curve_fit(sigmoid, self.X_test.iloc[:, 1], self.y_test, p0=init_params)

        # Generate the fitted curve using estimated parameters
        x_range = np.linspace(min(self.X_test.iloc[:, 1]), max(self.X_test.iloc[:, 1]), 500)
        y_out = sigmoid(x_range, params[0], params[1])

        # Create an array of predicted probabilities using the fitted curve
        y_pred_fitted = sigmoid(self.X_test.iloc[:, 1], params[0], params[1])

        # Plot the scatter plot using the fitted sigmoid curve
        plt.figure(figsize=(10, 6))

        # Scatter plot for males (blue circles)
        plt.scatter(self.X_test_male.iloc[:, 1], y_pred_fitted[self.X_test['sex_M'] == 1], marker='o', alpha=0.5,
                    label='Male Data', color='blue')

        # Scatter plot for females (red crosses)
        plt.scatter(self.X_test_female.iloc[:, 1], y_pred_fitted[self.X_test['sex_M'] == 0], marker='x', alpha=0.5,
                    label='Female Data', color='red')

        plt.xlabel('Predictor')
        plt.ylabel('Predicted Probability')
        plt.title('Fitted Sigmoid Curve with Male and Female Data')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def calculate_accuracy(self):
        predicted_classes = np.where(self.y_prob > 0.5, 1, 0)
        accuracy = np.mean(predicted_classes == self.y_test)
        # Print the accuracy as a percentage
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        return accuracy

    def calculate_accuracy_(self):
        y_pred = np.where(self.y_prob >= 0.5, 1, 0)
        accuracy = accuracy_score(self.y_test, y_pred)
        return [accuracy, y_pred]

    def calculate_confusion_matrix(self):
        y_pred = np.where(self.y_prob >= 0.5, 1, 0)
        cm = confusion_matrix(self.y_test, y_pred)
        return cm

    def print_evaluation_metrics(self):
        acc_pre = self.calculate_accuracy_()
        accuracy = acc_pre[0]
        y_pred = acc_pre[1]
        cm = self.calculate_confusion_matrix()
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print("Confusion Matrix:")
        print(cm)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)


# Load the Titanic dataset from seaborn library
titanic = sns.load_dataset("titanic")

# Perform Logistic Regression Analysis
logreg_analysis = LogisticRegressionAnalysis(titanic)


'''               Actual Positive    Actual Negative
Predicted Positive      TP                FP
Predicted Negative      FN                TN
'''

'''
Accuracy: (TP + TN) / Total
Precision: TP / (TP + FP)
Recall (Sensitivity or True Positive Rate): TP / (TP + FN)
Specificity (True Negative Rate): TN / (TN + FP)
F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
False Positive Rate: FP / (FP + TN)
'''