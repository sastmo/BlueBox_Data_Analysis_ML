# https://www.predictiveresearchsolutions.com/post/data-science-tips-feature-selection-using-boruta-in-python
# https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
import scipy as sp

# Create a sample dataset
X = pd.DataFrame({
    'age': [25, 32, 47, 51, 62],
    'height': [182, 176, 174, 168, 181],
    'weight': [75, 71, 78, 72, 86]
})

y = pd.Series([20, 32, 45, 55, 61], name='income')

# Create shadow features
np.random.seed(42)
X_shadow = X.apply(np.random.permutation)
X_shadow.columns = ['shadow_' + feat for feat in X.columns]
print("Shadow Features:\n", X_shadow)

# Combine original and shadow features
X_boruta = pd.concat([X, X_shadow], axis=1)
print("\nCombined Features (X_boruta):\n", X_boruta)

# Fit a random forest model
forest = RandomForestRegressor(max_depth=5, random_state=42, n_estimators=100)
forest.fit(X_boruta, y)

# Store and print feature importances
feat_imp_X = forest.feature_importances_[:len(X.columns)]
print("\nFeature VIM = ", feat_imp_X)
feat_imp_shadow = forest.feature_importances_[len(X.columns):]
print("\nShadow VIM = ", feat_imp_shadow)

# Compute shadow threshold and hits
shadow_threshold = round(feat_imp_shadow.max(), 3)
print("\nShadow Threshold = ", shadow_threshold)
hits = feat_imp_X > shadow_threshold
print("\nHits = ", hits)

# Create a DataFrame to accumulate hits over iterations
hits_counter = np.zeros((len(X.columns)))

# Repeat the process 20 times
for iter_ in range(20):
    # Create shadow features
    np.random.seed(iter_)
    X_shadow = X.apply(np.random.permutation)
    X_boruta = pd.concat([X, X_shadow], axis=1)

    # Fit a random forest model
    forest = RandomForestRegressor(max_depth=5, random_state=42, n_estimators=100)
    forest.fit(X_boruta, y)

    # Store feature importance
    feat_imp_X = forest.feature_importances_[:len(X.columns)]
    feat_imp_shadow = forest.feature_importances_[len(X.columns):]

    # Calculate hits for this trial and add to the counter
    hits_counter += (feat_imp_X > feat_imp_shadow.max())

# Create a DataFrame to display total hits over iterations
hits_df = pd.DataFrame({'var': X.columns, 'total hits in iteration': hits_counter})
print("\nTotal Hits Over Iterations:\n", hits_df)

# Calculate and plot the probability mass function using binomial distribution
trials = 20
pmf = [sp.stats.binom.pmf(x, trials, 0.5) for x in range(trials + 1)]

# Plot the probability mass function
pyplot.plot(list(range(0, trials + 1)), pmf, color="black")

# Visualize hits for age, weight, and height
pyplot.axvline(hits_df.loc[hits_df['var'] == 'age', 'total hits in iteration'].values, color='green')
pyplot.text(hits_df.loc[hits_df['var'] == 'age', 'total hits in iteration'].values - 1.5, 0.1, 'age')
pyplot.axvline(hits_df.loc[hits_df['var'] == 'weight', 'total hits in iteration'].values, color='red')
pyplot.text(hits_df.loc[hits_df['var'] == 'weight', 'total hits in iteration'].values + 1, 0.1, 'weight')
pyplot.axvline(hits_df.loc[hits_df['var'] == 'height', 'total hits in iteration'].values, color='gray')
pyplot.text(hits_df.loc[hits_df['var'] == 'height', 'total hits in iteration'].values + 1, 0.1, 'height')

# Show the plot
pyplot.show()
