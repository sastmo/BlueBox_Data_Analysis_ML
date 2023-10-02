import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pdpbox import pdp
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import make_scorer


class GradientBoostingModel:
    def __init__(self, random_seed=0):
        self.random_seed = random_seed
        self.best_params_ = None
        self.best_alpha = None
        self.best_reg = None

    # Define a custom scoring function that calculates the Huber loss with the best alpha
    def huber_loss_score(self, y_true, y_pred):
        alpha = self.best_alpha
        error = y_true - y_pred
        huber_loss = np.where(np.abs(error) < alpha, 0.5 * error ** 2, alpha * (np.abs(error) - 0.5 * alpha))
        return 1 - np.mean(huber_loss)

    def evaluate(self, X_train, y_train, X_test, y_test):
        # Create a custom scoring function with the best alpha
        custom_scorer = make_scorer(self.huber_loss_score)

        # Evaluate the best model on training data
        train_score = custom_scorer(self.best_reg, X_train, y_train)

        # Evaluate the best model on testing data
        test_score = custom_scorer(self.best_reg, X_test, y_test)
        print(f'Huber loss (train): {train_score:.4f}')
        print(f'Huber loss (test): {test_score:.4f}')

    def tuning_learning_curve(self, X_train, y_train, X_test, y_test):

        # Train a GradientBoostingRegressor with default hyperparameters
        reg = GradientBoostingRegressor(random_state=self.random_seed)
        reg.fit(X_train, y_train)

        # Evaluate the default model
        train_score = reg.score(X_train, y_train)
        test_score = reg.score(X_test, y_test)
        print(f'R2 score (train): {train_score:.4f}')
        print(f'R2 score (test): {test_score:.4f}')

        # Define hyperparameters for randomized search
        params = {
            'learning_rate': [0.1, 0.05, 0.02, 0.01],
            'n_estimators': [10, 50, 100, 500, 1000],
            'max_depth': [3, 6],
            'min_samples_leaf': [3, 5, 7],
            'subsample': [0.5, 1.0, 0.1],
            'max_features': [1.0, 0.3, 0.1],
            'loss': ['ls', 'huber'],  # Include Huber loss
            'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]  # Tuning alpha for Huber loss
        }

        # Perform randomized search for hyperparameter tuning
        search = RandomizedSearchCV(GradientBoostingRegressor(random_state=self.random_seed),
                                    params, n_iter=50, cv=3, n_jobs=-1)
        search.fit(X_train, y_train)
        self.best_params_ = search.best_params_

        # Print the best hyperparameters
        print(f"The best hyperparameter using Random Search:", self.best_params_)

        # Extract the best alpha value from best_params_
        self.best_alpha = self.best_params_['alpha']

        # Get the best model from the search
        self.best_reg = GradientBoostingRegressor(**self.best_params_)
        self.best_reg.fit(X_train, y_train)

        # Evaluate the best model on both training and testing data
        self.evaluate(X_train, y_train, X_test, y_test)

        # Calculate and plot learning curve with Huber Loss
        test_score = np.zeros(self.best_reg.n_estimators_)
        for i, y_test_pred in enumerate(self.best_reg.staged_predict(X_test)):
            alpha = self.best_alpha  # Use the best alpha for Huber Loss
            error = y_test - y_test_pred
            huber_loss = np.where(np.abs(error) < alpha, 0.5 * error ** 2, alpha * (np.abs(error) - 0.5 * alpha))
            test_score[i] = np.mean(huber_loss)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.best_reg.n_estimators), self.best_reg.train_score_, label='Training loss')
        plt.plot(np.arange(self.best_reg.n_estimators), test_score, 'r', label='Test loss (Huber Loss)')
        plt.xlabel('Boosting iterations')
        plt.ylabel('Huber Loss')
        plt.legend()
        plt.show()

    def perform_search_early_stopping(self,  X_train, y_train, X_test, y_test):

        # Perform a second randomized search with a fixed number of estimators using Early Stopping
        params = {
            'learning_rate': [0.1, 0.05, 0.02, 0.01],
            'max_depth': [3, 7],
            'min_samples_leaf': [3, 5, 9],
            'subsample': [0.5, 1.0, 0.1],
            'max_features': [1.0, 0.3, 0.1],
            'loss': ['ls', 'huber'],  # Include Huber loss
            'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]  # Tuning alpha for Huber loss
        }

        search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=self.random_seed, n_estimators=300, n_iter_no_change=5),
            params, n_iter=50, cv=3, n_jobs=-1
        )
        search.fit(X_train, y_train)

        self.best_params_ = search.best_params_

        # Get the best model from the search
        self.best_reg = GradientBoostingRegressor(**self.best_params_)
        self.best_reg.fit(X_train, y_train)

        # Number of tree estimators
        numb_tree = self.best_reg.n_estimators_
        print(f"Number of tree estimator based on early stopping method is {numb_tree}")

        # Evaluate the best model on both training and testing data
        self.evaluate(X_train, y_train, X_test, y_test)

    def feature_importance(self, X_train, names):
        # Sort the features by their importance and plot
        feature_importance = self.best_reg.feature_importances_
        sorted_idx = np.argsort(feature_importance)

        plt.figure(figsize=(10, 6))
        pos = np.arange(len(feature_importance))
        plt.barh(pos, feature_importance[sorted_idx])
        plt.yticks(pos, np.array(names)[sorted_idx])
        plt.xlabel('Feature importance')
        plt.show()

    def plot_partial_dependence(self, X_train, initial_features):
        # Create a list to store the final features including indices
        features = []

        # Extract column indices based on feature names and convert names to indices
        for feature in initial_features:
            if isinstance(feature, tuple):
                # If it's a tuple, map feature names to indices
                indices = [X_train.columns.get_loc(item) for item in feature]
                features.append(tuple(indices))
            else:
                # If it's a name, find the index and add both the name and index
                index = X_train.columns.get_loc(feature)
                features.append(index)

        # Generate the partial dependence plot
        pdp_display = PartialDependenceDisplay.from_estimator(self.best_reg, X_train, features, target=0)

        # Set the feature names for display
        pdp_display.feature_names = X_train.columns  # Use X.columns to set feature names

        # Display the plot using matplotlib
        pdp_display.plot()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Load the California housing dataset
    data = fetch_california_housing()

    # Create a Pandas DataFrame from the dataset
    california_df = pd.DataFrame(data.data, columns=data.feature_names)

    # Add the target column to the DataFrame
    california_df['target'] = data.target

    # Split the data into features (X) and target (y)
    X = california_df.drop(columns=['target'])  # Features
    y = california_df['target']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create an instance of the GradientBoostingModel class
    gb_model = GradientBoostingModel(random_seed=0)

    # Fit the model and perform the second randomized search
    # gb_model.tuning_learning_curve(X_train, y_train, X_test, y_test)
    gb_model.perform_search_early_stopping(X_train, y_train, X_test, y_test)

    # Plot feature importance
    gb_model.feature_importance(X_train, X_train.columns)

    # Define the initial features for partial dependence
    initial_features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms', ('AveOccup', 'HouseAge')]

    # Generate and plot partial dependence
    gb_model.plot_partial_dependence(X_train, initial_features)
