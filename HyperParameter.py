# https://medium.com/@ali.soleymani.co/stop-using-grid-search-or-random-search-for-hyperparameter-tuning-c2468a2ff887

# Import necessary libraries
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

# Reading the dataset
loan_data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\day.csv")

# Prepare the data
y = loan_data['cnt']
X = loan_data.drop(['dteday', 'cnt'], axis=1)

# Method 1: Grid Search


# Split the data without stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=38)

# Creating an XGBoost regressor instance
xgb_tuned = XGBRegressor(random_state=1)

# Defining a grid of parameters to be explored
parameters = {
    "n_estimators": [10, 50, 100],
    "subsample": [0.6, 0.8, 1],
    "learning_rate": [0.01, 0.1, 0.5, 1],
    "gamma": [0.01, 0.1, 1, 5],
    "colsample_bytree": [0.5, 0.7, 0.9, 1],
    "alpha": [0, 0.1, 0.5]
}

# Choosing the scoring metric for comparing parameter combinations
scorer = metrics.make_scorer(metrics.r2_score)

# Running the Grid Search
grid_obj = GridSearchCV(xgb_tuned, parameters, scoring=scorer,
                        cv=5, n_jobs=-1, verbose=2)

grid_obj.fit(X_train, y_train)

# Printing the best parameters from Grid Search
print("Best parameters from Grid Search:")
print(grid_obj.best_params_)

# Method 2: Randomized Search

# Import necessary library for Randomized Search
from sklearn.model_selection import RandomizedSearchCV

# Running the Randomized Search
rand_obj = RandomizedSearchCV(xgb_tuned, parameters, scoring=scorer,
                              n_iter=20, n_jobs=-1, cv=5, verbose=1)

rand_obj.fit(X_train, y_train)

# Printing the best parameters from Randomized Search
print("Best parameters from Randomized Search:")
print(rand_obj.best_params_)


# Method 3: Bayesian Search on the same search space as Grid Search

# Import necessary libraries
from xgboost import XGBRegressor
from skopt import BayesSearchCV

# Creating an XGBoost regressor instance
xgb_tuned = XGBRegressor(random_state=1)

# Defining a grid of parameters for Bayesian Search
parameters = {
    "n_estimators": [10, 50, 100],
    "subsample": [0.6, 0.8, 1],
    "learning_rate": [0.01, 0.1, 0.5, 1],
    "gamma": [0.01, 0.1, 1, 5],
    "colsample_bytree": [0.5, 0.7, 0.9, 1],
    "alpha": [0, 0.1, 0.5]
}

# Applying Bayesian Search
bayes = BayesSearchCV(xgb_tuned, search_spaces=parameters, n_iter=20, cv=5)
bayes.fit(X_train, y_train)

# Displaying the best hyperparameters found
print("Best hyperparameters from Bayesian Search on the same space:")
print(bayes.best_params_)

# Method 4: Using continuous Search Spaces for Bayesian Search

# Import necessary libraries
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize

# Defining a continuous search space for Bayesian Search
space = [Integer(1, 20, name='max_depth'),
         Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
         Real(0.5, 1, "uniform", name='subsample'),
         Real(10**-5, 10**1, "uniform", name='gamma'),
         Real(10**-5, 10**0, "uniform", name='alpha')]

# Defining the objective function for Bayesian Search
@use_named_args(space)
def objective(**params):
    xgb_tuned.set_params(**params)
    return -np.mean(cross_val_score(xgb_tuned, X, y, cv=5, n_jobs=-1,
                                     scoring="neg_mean_absolute_error"))

# Applying Bayesian Search with continuous search space
res_gp = gp_minimize(objective, space, n_calls=20, random_state=0)

# Displaying the best hyperparameters found
print("Best hyperparameters from Bayesian Search on continuous space:")
print(res_gp.x)
