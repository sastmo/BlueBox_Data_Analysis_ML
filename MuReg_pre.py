# https://chat.openai.com/share/edf22c9b-5630-4386-bef0-e74e8a4bbe0f
# https://www.fireblazeaischool.in/blogs/assumptions-of-linear-regression/
# https://datatab.net/statistics-calculator/regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
california_data = fetch_california_housing(as_frame=True)
california = california_data.frame

# Select relevant features and target
X = california[['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']]
y = california['AveBedrms']

# Assumption 1: Linear Relationship
# Scatter plots to visualize linear relationship
plt.figure(figsize=(15, 5))
for i, feature in enumerate(X.columns):
    plt.subplot(1, 4, i+1)
    plt.scatter(X[feature], y)
    plt.xlabel(feature)
    plt.ylabel('Average Bedrooms')
    plt.title(f'{feature} vs. AveBedrms')

plt.tight_layout()
plt.show()

# Assumption 2: No or Low Multicollinearity
vif = pd.DataFrame()
vif['Feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

# Assumption 3: Homoscedasticity
X_constant = sm.add_constant(X)
model = sm.OLS(y, X_constant).fit()
plt.scatter(model.fittedvalues, model.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Assumption 4: No Autocorrelation
dw_statistic = durbin_watson(model.resid)
print(f'Durbin-Watson statistic: {dw_statistic}')

# Assumption 5: Normality of Residuals
sm.qqplot(model.resid, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Assumption 6: Independent Residuals
plt.scatter(np.arange(len(model.resid)), model.resid)
plt.xlabel('Order of Data Collection')
plt.ylabel('Residuals')
plt.title('Residuals vs. Order of Data Collection')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Assumption 7: No Perfect Collinearity
# Already checked along with multicollinearity

# Fit the multiple regression model
model = sm.OLS(y, X_constant).fit()

# Print the summary
print(model.summary())
