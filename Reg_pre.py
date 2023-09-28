# https://chat.openai.com/share/edf22c9b-5630-4386-bef0-e74e8a4bbe0f
# https://www.fireblazeaischool.in/blogs/assumptions-of-linear-regression/
# https://ademos.people.uic.edu/Chapter12.html
# https://datatab.net/statistics-calculator/regression


from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# Load the California Housing dataset
housing = fetch_california_housing()
X = housing.data  # Features
y = housing.target  # Target

# Assumption 1: Linear Relationship
plt.scatter(X[:, 0], y)
plt.xlabel('Feature 0')
plt.ylabel('Target')
plt.title('Assumption 1: Linear Relationship')
plt.show()

# Assumption 2: No Autocorrelation
X_with_constant = sm.add_constant(X)
model = sm.OLS(y, X_with_constant).fit()
dw_statistic = durbin_watson(model.resid)
print(f'Assumption 2 - Durbin-Watson statistic: {dw_statistic}')

# Assumption 3: Multivariate Normality
model = sm.OLS(y, X_with_constant).fit()
residuals = model.resid
sm.qqplot(residuals, line='s')
plt.title('Assumption 3: Multivariate Normality')
plt.show()

# Assumption 4: Homoscedasticity
plt.scatter(model.fittedvalues, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Assumption 4: Homoscedasticity')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Assumption 5: No or Low Multicollinearity
vif = pd.DataFrame()
vif['Feature'] = housing.feature_names
vif['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print(vif)

# Assumption 6: Homoscedasticity (Using the Ames housing dataset)
data = pd.DataFrame({'X': X[:, 0], 'Y': y})
model = sm.OLS(data['Y'], sm.add_constant(data['X'])).fit()
residuals = model.resid
plt.scatter(model.predict(), residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Assumption 6: Homoscedasticity')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Assumption 7: Independence of Residuals
plt.scatter(np.arange(len(residuals)), residuals)
plt.xlabel('Order of Data Collection')
plt.ylabel('Residuals')
plt.title('Assumption 7: Independence of Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Assumption 8: Normality of Residuals
sm.qqplot(residuals, line='s')
plt.title('Assumption 8: Normality of Residuals')
plt.show()

# Fit the linear regression model
X_with_constant = sm.add_constant(data['X'])
model = sm.OLS(data['Y'], X_with_constant).fit()

# Print the summary
print(model.summary())
