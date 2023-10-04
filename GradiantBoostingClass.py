"""
This script introduces the GradientBoostingModel class, designed to facilitate Gradient Boosting regression tasks.
The class provides the following methods:

evaluate: Assess model performance and visualize it with a validation curve.
tuning_learning_curve: Perform hyperparameter tuning and evaluate model performance.
perform_search_early_stopping: Optimize hyperparameters with early stopping and analyze feature importance.
plot_partial_dependence: Generate partial dependence plots to understand feature-target relationships.
The script simplifies the process of building and fine-tuning Gradient Boosting models for various regression tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import validation_curve, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay


class GradientBoostingModel:
    """
    A class for performing Gradient Boosting regression and hyperparameter tuning.
    """

    def __init__(self, random_seed=0):
        """
        Initialize the GradientBoostingModel.

        Args:
            random_seed (int): Random seed for reproducibility.
        """
        self.random_seed = random_seed
        self.best_params_ = None
        self.best_alpha = None
        self.best_reg = None

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the GradientBoostingRegressor model and plot a validation curve.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training target.
            X_test (array-like): Testing features.
            y_test (array-like): Testing target.
        """
        # Custom loss function (Huber loss)
        def custom_huber_loss(y_true, y_pred, alpha):
            # Huber loss calculation
            error = y_true - y_pred
            huber_loss = np.where(np.abs(error) < alpha, 0.5 * error ** 2, alpha * (np.abs(error) - 0.5 * alpha))
            return np.mean(huber_loss)

        # Create and fit the GradientBoostingRegressor model with the best parameters
        best_reg = GradientBoostingRegressor(**self.best_params_)
        best_reg.fit(X_train, y_train)

        # Predict on the training and testing data
        y_train_pred = best_reg.predict(X_train)
        y_test_pred = best_reg.predict(X_test)

        # Evaluate the best model on training data using the Huber loss
        train_loss = custom_huber_loss(y_train, y_train_pred, alpha=self.best_alpha)

        # Evaluate the best model on testing data using the Huber loss
        test_loss = custom_huber_loss(y_test, y_test_pred, alpha=self.best_alpha)

        # Evaluate the best model on training and testing data using the R^2 score
        train_score = best_reg.score(X_train, y_train)
        test_score = best_reg.score(X_test, y_test)

        # Print the mean values of the losses and scores
        print(f'Huber Loss (train): {np.mean(train_loss):.4f}')
        print(f'Huber Loss (test): {np.mean(test_loss):.4f}')
        print(f'R^2 Score (train): {np.mean(train_score):.4f}')
        print(f'R^2 Score (test): {np.mean(test_score):.4f}')

        # Create a range of values for the n_estimators parameter
        param_range = np.arange(1, 1001, 10)

        # Plot the validation curve for the model using different values of n_estimators
        train_scores, test_scores = validation_curve(GradientBoostingRegressor(**self.best_params_),
                                                     X_train, y_train, param_name='n_estimators',
                                                     param_range=param_range, scoring='r2')
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, 1 - np.mean(train_scores, axis=1), label='Training score')
        plt.plot(param_range, 1 - np.mean(test_scores, axis=1), label='Test score')
        plt.xlabel('Number of trees')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

    def tuning_learning_curve(self, X_train, y_train, X_test, y_test):
        """
        Perform hyperparameter tuning and evaluate using a learning curve.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training target.
            X_test (array-like): Testing features.
            y_test (array-like): Testing target.
        """
        # Create an instance of StandardScaler
        scaler = StandardScaler()

        # Fit and transform the training data using the scaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        # Train a GradientBoostingRegressor with default hyperparameters
        reg = GradientBoostingRegressor(random_state=self.random_seed)
        reg.fit(X_train, y_train)

        # Evaluate the default model
        train_score = reg.score(X_train_scaled, y_train)
        test_score = reg.score(X_test_scaled, y_test)

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
        search.fit(X_train_scaled, y_train)
        self.best_params_ = search.best_params_

        # Print the best hyperparameters
        print(f"The best hyperparameter using Random Search:", self.best_params_)

        # Extract the best alpha value from best_params_
        self.best_alpha = self.best_params_['alpha']

        # Get the best model from the search
        self.best_reg = GradientBoostingRegressor(**self.best_params_)
        self.best_reg.fit(X_train, y_train)

        # Evaluate the best model on both training and testing data
        self.evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

    def perform_search_early_stopping(self, X_train, y_train, X_test, y_test, names):
        """
        Perform hyperparameter tuning with early stopping and evaluate the model.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training target.
            X_test (array-like): Testing features.
            y_test (array-like): Testing target.
            names (list): List of feature names.
        """
        # Create an instance of StandardScaler
        scaler = StandardScaler()

        # Fit and transform the training data using the scaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        # Perform a second randomized search with a fixed number of estimators using Early Stopping
        params = {
            'learning_rate': [0.1, 0.05, 0.02, 0.01],
            'max_depth': [3, 6],
            'min_samples_leaf': [3, 5, 9],
            'subsample': [0.5, 1.0, 0.1],
            'max_features': [1.0, 0.3, 0.1],
            'loss': ['ls', 'huber'],  # Include Huber loss
            'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]  # Tuning alpha for Huber loss
        }

        search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=self.random_seed, n_estimators=1000, n_iter_no_change=10),
            params, n_iter=50, cv=3, n_jobs=-1
        )
        search.fit(X_train_scaled, y_train)

        self.best_params_ = search.best_params_

        # Print the best hyperparameters
        print(f"The best hyperparameter using Random Search:", self.best_params_)

        # Extract the best alpha value from best_params_
        self.best_alpha = self.best_params_['alpha']

        # Get the best model from the search
        self.best_reg = GradientBoostingRegressor(**self.best_params_)
        self.best_reg.fit(X_train_scaled, y_train)

        # Number of tree estimators
        numb_tree = self.best_reg.n_estimators_
        print(f"Number of tree estimator based on early stopping method is {numb_tree}")

        # Evaluate the best model on both training and testing data
        self.evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

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
        """
        Plot partial dependence plots for selected features.

        Args:
            X_train (array-like): Training features.
            initial_features (list): List of selected features or feature combinations.
        """
        # Create an instance of StandardScaler
        scaler = StandardScaler()

        # Fit and transform the training data using the scaler
        X_train_scaled = scaler.fit_transform(X_train)

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
        pdp_display = PartialDependenceDisplay.from_estimator(self.best_reg, X_train_scaled, features, target=0)

        # Set the feature names for display
        pdp_display.feature_names = X_train.columns  # Use X.columns to set feature names

        # Display the plot using matplotlib
        pdp_display.plot()
        plt.show()
