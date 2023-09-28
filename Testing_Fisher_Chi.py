import scipy.stats as stats

# Fisher's Exact Test *****************

'''         | Process A | Process B | Total
-----------------------------------------
Defect   |    15     |    3      |  13
No Defect|    85     |    97     |  187
-----------------------------------------
Total    |   100     |   100     |  200
'''

# Create the 2x2 contingency table
contingency_table = [[15, 3], [80, 97]]

# Perform Fisher's Exact Test
odds_ratio, p_value = stats.fisher_exact(contingency_table)

print("Odds Ratio:", odds_ratio)
print("p-value:", p_value)

if p_value < 0.05:
    print("There is a significant association between process and defect status.")
else:
    print("There is no significant association between process and defect status.")


'''         | Treatment A | Treatment B | Total
-------------------------------------------
Recovered|     15      |     5       |  20
Not Recover|     3       |     12      |  15
-------------------------------------------
Total    |     18      |     17      |  35
'''

import scipy.stats as stats

# Create the 2x2 contingency table
contingency_table = [[15, 5], [3, 12]]

# Perform Fisher's Exact Test
odds_ratio, p_value = stats.fisher_exact(contingency_table)

print("Odds Ratio:", odds_ratio)
print("p-value:", p_value)

if p_value < 0.05:
    print("There is a significant association between treatment and recovery.")
else:
    print("There is no significant association between treatment and recovery.")


print("_______________________________")

# Chi-Squared Test *********************

'''|          | Apple | Samsung | Google |
|----------|-------|---------|--------|
| 18-30    |  40   |   30    |   15   |
| 31-50    |  50   |   20    |   10   |
| 51+      |  20   |   10    |   5    |
'''

import numpy as np
import scipy.stats as stats

# Create the contingency table
observed = np.array([[40, 30, 15],
                     [50, 20, 10],
                     [20, 10, 5]])

# Perform Chi-Squared Test without continuity correction
chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=False)

print("Without Continuity Correction:")
print("Chi-Squared:", chi2)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)
print("p-value:", p_value)

if p_value < 0.05:
    print("There is a significant association between smartphone brand and age group.")
else:
    print("There is no significant association between smartphone brand and age group.")

print("\n")

# Perform Chi-Squared Test with continuity correction
chi2_corr, p_value_corr, dof_corr, expected_corr = stats.chi2_contingency(observed, correction=True)

print("With Continuity Correction:")
print("Chi-Squared:", chi2_corr)
print("Degrees of Freedom:", dof_corr)
print("Expected Frequencies:\n", expected_corr)
print("p-value:", p_value_corr)

if p_value_corr < 0.05:
    print("There is a significant association between smartphone brand and age group.")
else:
    print("There is no significant association between smartphone brand and age group.")

print("_______________________________")

# Wald Test ***************************

import pandas as pd
import statsmodels.api as sm

# Create a dataset
data = {
    'Hours_Study': [5, 8, 10, 4, 6, 9, 7, 3, 2, 11],
    'Pass_Exam': [0, 1, 1, 0, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# Add an intercept column
df['Intercept'] = 1

# Perform logistic regression with increased max_iter
X = df[['Intercept', 'Hours_Study']]
y = df['Pass_Exam']
model = sm.Logit(y, X).fit(maxiter=100)  # Adjust the maxiter parameter

# Summary of the logistic regression model
print(model.summary())

# Perform the Wald Test
test_result = model.wald_test('Hours_Study = 0')
print("Wald Test p-value:", test_result.pvalue)

# Check if the coefficient for Hours_Study is significant
significance_level = 0.05
if test_result.pvalue < significance_level:
    print("The coefficient for Hours_Study is statistically significant.")
else:
    print("The coefficient for Hours_Study is not statistically significant.")

