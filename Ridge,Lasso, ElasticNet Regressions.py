# https://github.com/free-to-learn/Machine-Learning-Concepts/blob/master/Lasso%20%2C%20Ridge%20and%20ElasticNet%20%20Regression.ipynb
# https://chat.openai.com/share/610595e1-4c25-45c3-9603-dc1c66022999
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


class DataPreprocessor:
    def __init__(self, file_path, header_names, missing_values):
        self.file_path = file_path
        self.header_names = header_names
        self.missing_values = missing_values

    def load_data(self):
        housing_data = pd.read_csv(self.file_path, header=None, delimiter="\s+", engine="python",
                                   na_values=self.missing_values)
        df = pd.DataFrame(housing_data.values, columns=self.header_names)
        X = df.iloc[:, :-1]
        y = df['MEDV']
        return X, y


class RidgeRegressionAnalyzer:
    def __init__(self, alphas, X_train, y_train, X_test, y_test):
        self.alphas = alphas
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def calculate_best_alpha(self):
        cv_scores = []

        for alpha in self.alphas:
            rr = Ridge(alpha=alpha)
            scores = cross_val_score(rr, self.X_train, self.y_train, cv=6)
            cv_scores.append(np.mean(scores))

        best_alpha_idx = np.argmax(cv_scores)
        return self.alphas[best_alpha_idx]

    def analyze_ridge(self, alpha):
        rr = Ridge(alpha=alpha)
        rr.fit(self.X_train, self.y_train)

        ridge_coef = rr.coef_
        ridge_intercept = rr.intercept_
        ridge_train_score = rr.score(self.X_train, self.y_train)
        ridge_test_score = rr.score(self.X_test, self.y_test)

        return ridge_coef, ridge_intercept, ridge_train_score, ridge_test_score


class LassoRegressionAnalyzer:
    def __init__(self, alphas, X_train, y_train, X_test, y_test):
        self.alphas = alphas
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def calculate_best_alpha(self):
        cv_scores = []

        for alpha in self.alphas:
            lasso = Lasso(alpha=alpha)  # Corrected 'lasso' to 'Lasso'
            scores = cross_val_score(lasso, self.X_train, self.y_train, cv=6)
            cv_scores.append(np.mean(scores))

        best_alpha_idx = np.argmax(cv_scores)
        return self.alphas[best_alpha_idx]

    def analyze_lasso(self, alpha):
        lasso = Lasso(alpha=alpha)
        lasso.fit(self.X_train, self.y_train)

        lasso_coef = lasso.coef_
        lasso_intercept = lasso.intercept_
        lasso_train_score = lasso.score(self.X_train, self.y_train)
        lasso_test_score = lasso.score(self.X_test, self.y_test)

        return lasso_coef, lasso_intercept, lasso_train_score, lasso_test_score


class ElasticNetRegressionAnalyzer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def analyze_elastic_net(self):
        enet = ElasticNet()
        enet.fit(self.X_train, self.y_train)

        enet_coef = enet.coef_
        enet_train_score = enet.score(self.X_train, self.y_train)
        enet_test_score = enet.score(self.X_test, self.y_test)

        print("ElasticNet Coefficients:")
        for i, v in enumerate(enet_coef):
            print('Feature %2d: Score %.5f' % (i, v))

        plt.bar([x for x in range(len(enet_coef))], enet_coef)
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.title('ElasticNet Coefficients')
        plt.show()

        return enet_coef, enet_train_score, enet_test_score


class RegressionComparison:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fit_and_analyze(self):
        # LassoCV
        Lasso_CV = LassoCV()
        Lasso_CV.fit(self.X_train, self.y_train)

        # RidgeCV
        Ridge_CV = RidgeCV()
        Ridge_CV.fit(self.X_train, self.y_train)

        # ElasticNetCV
        ENETCV = ElasticNetCV()
        ENETCV.fit(self.X_train, self.y_train)

        # Obtain coefficients
        Lasso_CV_coef = Lasso_CV.coef_
        Ridge_CV_coef = Ridge_CV.coef_
        ENETCV_coef = ENETCV.coef_

        # Print and plot feature importance
        print("LassoCV Coefficients:")
        self.print_feature_importance(Lasso_CV_coef)

        print("\nRidgeCV Coefficients:")
        self.print_feature_importance(Ridge_CV_coef)

        print("\nElasticNetCV Coefficients:")
        self.print_feature_importance(ENETCV_coef)

        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'Lasso_CV_Pred': Lasso_CV.predict(self.X_test),
            'Ridge_Cv_Pred': Ridge_CV.predict(self.X_test),
            "Elastic_Net_cv": ENETCV.predict(self.X_test),
            # ... (Add other model predictions)
            "Actual_Data": self.y_test
        })

        # Create LassoCV predictions DataFrame
        LassoCV_Prediction = Lasso_CV.predict(self.X_test)
        lasso_prediction_df = pd.DataFrame({
            "LassoCV_Prediction": LassoCV_Prediction,
            "Actual": self.y_test,
            "Error": self.y_test - LassoCV_Prediction
        })

        return predictions, lasso_prediction_df


    def print_feature_importance(self, coef):
        for i, v in enumerate(coef):
            print('Feature: %0d, Score: %.5f' % (i, v))


    def plot_feature_importance(self, coef):
        plt.bar([x for x in range(len(coef))], coef)
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.title('Feature Importance')
        plt.show()


def main():
    file_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\housing.csv"
    header_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                    'MEDV']
    missing_values = ["?", "NA"]
    alphas = np.logspace(-50, 50, 13)

    data_preprocessor = DataPreprocessor(file_path, header_names, missing_values)
    X, y = data_preprocessor.load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ridge_analyzer = RidgeRegressionAnalyzer(alphas, X, y, X_test, y_test)
    best_alpha = ridge_analyzer.calculate_best_alpha()
    print("Best alpha:", best_alpha)

    elastic_net_analyzer = ElasticNetRegressionAnalyzer(X_train, y_train, X_test, y_test)
    elastic_net_coef, elastic_net_train_score, elastic_net_test_score = elastic_net_analyzer.analyze_elastic_net()

    print("ElasticNet Train Score:", elastic_net_train_score)
    print("ElasticNet Test Score:", elastic_net_test_score)

    ridge_coef, ridge_intercept, ridge_train_score, ridge_test_score = ridge_analyzer.analyze_ridge(best_alpha)

    print("Ridge Intercept (alpha = Best alpha):", ridge_intercept)
    print("Ridge Coefficients (alpha = ridge_coef):", ridge_coef)
    print("Ridge Train Score (alpha = Best alpha):", ridge_train_score)
    print("Ridge Test Score (alpha = Best alpha):", ridge_test_score)

    lasso_analyzer = LassoRegressionAnalyzer(alphas, X, y, X_test, y_test)
    best_alpha_lasso = lasso_analyzer.calculate_best_alpha()
    print("Best alpha for Lasso:", best_alpha_lasso)

    lasso_coef, lasso_intercept, lasso_train_score, lasso_test_score = lasso_analyzer.analyze_lasso(best_alpha_lasso)

    plt.figure(figsize=(16, 9))

    rr = Ridge(alpha=0.01)
    rr.fit(X_train, y_train)
    plt.plot(rr.coef_, alpha=0.7, linestyle='none', marker='*', markersize=15, color='red',
             label=r'Ridge; $\alpha = 0.01$', zorder=7)

    rr100 = Ridge(alpha=100)
    rr100.fit(X_train, y_train)
    plt.plot(rr100.coef_, alpha=0.5, linestyle='none', marker='d', markersize=8, color='blue',
             label=r'Ridge; $\alpha = 100$')

    rrbest_alpha = Ridge(alpha=best_alpha)
    rrbest_alpha.fit(X_train, y_train)
    plt.plot(rrbest_alpha.coef_, alpha=0.9, linestyle='none', marker='o', markersize=12, color='black',
             label=r'Ridge; $\alpha =$' + f'{best_alpha:.4f}')

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    plt.plot(lr.coef_, alpha=0.4, marker='o', markersize=17, color='green', label='Linear Regression')

    plt.xlabel('Coefficient Index', fontsize=16)
    plt.ylabel('Coefficient Magnitude', fontsize=16)
    plt.legend(fontsize=13, loc=4)
    plt.show()

    print("-_-" * 37)
    print("Creating a Linear Regression model")
    lr = LinearRegression()

    print("Fitting the model to the training data")
    lr.fit(X_train, y_train)

    coefficients = lr.coef_
    print("Coefficients:", coefficients)

    intercept = lr.intercept_
    print("Intercept:", intercept)

    train_score = lr.score(X_train, y_train)
    print("Training Score:", train_score)

    test_score = lr.score(X_test, y_test)
    print("Test Score:", test_score)

    print("Creating a Ridge Regression model with low alpha (0.01)")
    rr = Ridge(alpha=0.01)
    rr.fit(X_train, y_train)

    rr_coef = rr.coef_
    rr_intercept = rr.intercept_
    print("Ridge Coefficients (alpha = 0.01):", rr_coef)
    print("Ridge Intercept (alpha = 0.01):", rr_intercept)

    rr_train_score = rr.score(X_train, y_train)
    rr_test_score = rr.score(X_test, y_test)
    print("Ridge Train Score (alpha = 0.01):", rr_train_score)
    print("Ridge Test Score (alpha = 0.01):", rr_test_score)

    print("Creating a Ridge Regression model with high alpha (100)")
    rr100 = Ridge(alpha=100)
    rr100.fit(X_train, y_train)

    rr100_coef = rr100.coef_
    rr100_intercept = rr100.intercept_
    print("Ridge Coefficients (alpha = 100):", rr100_coef)
    print("Ridge Intercept (alpha = 100):", rr100_intercept)

    rr100_train_score = rr100.score(X_train, y_train)
    rr100_test_score = rr100.score(X_test, y_test)
    print("Ridge Train Score (alpha = 100):", rr100_train_score)
    print("Ridge Test Score (alpha = 100):", rr100_test_score)

    print("Ridge Intercept (alpha = Best alpha):", ridge_intercept)
    print("Ridge Coefficients (alpha = ridge_coef):", ridge_coef)
    print("Ridge Train Score (alpha = Best alpha):", ridge_train_score)
    print("Ridge Test Score (alpha = Best alpha):", ridge_test_score)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    regression_analyzer = RegressionComparison(X_train, y_train, X_test, y_test)
    predictions_df, lasso_prediction_df = regression_analyzer.fit_and_analyze()

    print("\nPredictions DataFrame:")
    print(predictions_df)

    print("\nLassoCV Predictions DataFrame:")
    print(lasso_prediction_df)


if __name__ == "__main__":
    main()
