import math
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the CSV file
titanic = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\titanic.csv")

# Create new columns in the DataFrame
titanic["New_age"] = titanic["Age"] / 25
titanic["New_col"] = titanic["Age"] / titanic["Fare"]

# Rename specific columns in the DataFrame
titanic_rename = titanic.rename(
    columns={
        "PassengerId": "Id",
        "New_col": "AF_ratio"
    }
)

# Display the first few rows of the DataFrame
print(titanic_rename.head(), end="\n")

# Calculate and display the average age of the Titanic passengers
avg_age_overall = titanic_rename["Age"].mean()
print("Avg age of the Titanic passenger is:", avg_age_overall, end="\n")

# Calculate and display the average age and fare together
avg_age_overall_1 = titanic_rename[["Age", "Fare"]].mean()
print(avg_age_overall_1, end="\n")

# Display summary statistics of the DataFrame
print(titanic_rename.describe())

# Calculate and display specific statistics for Age and Fare columns
cust_desc = titanic_rename.agg(
    {
        "Age": ["min", "max", "median"],
        "Fare": ["min", "max", "median"]
    }
)
print(cust_desc)

# Calculate and display the average age grouped by Sex
avg_age_sex = titanic_rename[["Sex", "Age"]].groupby('Sex').mean()  # ~ titanic_rename.groupby('Sex')["Age"].mean()
print(avg_age_sex)

# Calculate and display the total number of passengers
num_pass = titanic_rename["Pclass"].count()

# Calculate and display the number of passengers per Pclass using groupby and count
num_pss_clas = titanic_rename.groupby("Pclass")["Pclass"].count()

# Alternatively, calculate the number of passengers per Pclass using value_counts method
num_pss_clas1 = titanic_rename["Pclass"].value_counts()

# Another way to calculate the number of passengers per Pclass using groupby and size
num_pss_clas2 = titanic_rename.groupby("Pclass")["Pclass"].size()

# Display the results
print(num_pass, end="\n")
print(num_pss_clas, end="\n")
print(num_pss_clas1, end="\n")
print(num_pss_clas2, end="\n")

# Explanation of Nan handling in certain Pandas methods:

# siz -> nan = True (The siz method considers NaN values, so the result includes NaN counts)
# count -> nan = False (The count method does not consider NaN values in the count)
# value_counts -> nan = False (The value_counts method does not consider NaN values in the count)
# value_counts (dropna=False) -> nan = True (When dropna=False is used, value_counts considers NaN values)

# Importing pandas library
import pandas as pd

# Function to sort the DataFrame by "Pclass" and "Age" columns and display the top rows
titanic_rename = pd.DataFrame(...)  # Assuming you have a DataFrame called 'titanic_rename'
print("Sorted DataFrame by 'Pclass' and 'Age':")
print(titanic_rename.sort_values(by=["Pclass", "Age"]).head())
# The above code sorts the 'titanic_rename' DataFrame by "Pclass" in ascending order and then by "Age" in ascending order.
# It then prints the top rows of the sorted DataFrame.

# Reading the CSV file 'weather.csv' into a DataFrame named 'weather'
weather = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\weather.csv",
                      encoding="Unicode_escape", index_col="Date/Time", parse_dates=True)

# Filter the 'weather' DataFrame to include only rows where the 'Weather' column has the value 'Fog'
Wea_sub = weather[weather['Weather'] == 'Fog']

# Sort the filtered 'Wea_sub' DataFrame by index (Date/Time) and group by "Temp (C)" column
print("\nFoggy days sorted by temperature:")
print(Wea_sub.sort_index().groupby(["Temp (C)"]).head())
# The above code filters the 'weather' DataFrame to keep only rows with 'Weather' as 'Fog'.
# It then sorts the filtered DataFrame by index (Date/Time) and groups it by "Temp (C)".
# The sorted and grouped DataFrame is then printed.

# Create a pivot table with columns as "Temp (C)", values as "Stn Press (kPa)"
print("\nPivot table of 'Weather' and 'Temp (C)' with 'Stn Press (kPa)' as values:")
print(Wea_sub.pivot(columns='Temp (C)', values="Stn Press (kPa)"))
# The above code creates a pivot table using the 'Wea_sub' DataFrame.
# The pivot table has "Temp (C)" as columns, "Stn Press (kPa)" as values, and "Weather" as the index.

# Plot the pivot table
Wea_sub.pivot(columns='Temp (C)', values="Stn Press (kPa)").plot()
plt.show()
# The above code plots the pivot table created in the previous step using matplotlib.

# Create a pivot table with columns as "Weather", index as "Dew Point Temp (C)", and values as the mean of "Stn Press (kPa)"
print("\nPivot table of 'Weather', 'Dew Point Temp (C)' with mean of 'Stn Press (kPa)' as values:")
print(weather.pivot_table(values="Stn Press (kPa)", index="Dew Point Temp (C)", columns="Weather", aggfunc="mean"))
# The above code creates a pivot table using the 'weather' DataFrame.
# The pivot table has "Weather" as columns, "Dew Point Temp (C)" as index, and the mean of "Stn Press (kPa)" as values.

# Group the 'weather' DataFrame by "Weather" and "Dew Point Temp (C)" and calculate the mean
print("\nGrouped DataFrame with mean of 'Stn Press (kPa)':")
print(weather.groupby(["Weather", "Dew Point Temp (C)"]).mean())
# The above code groups the 'weather' DataFrame by "Weather" and "Dew Point Temp (C)" columns and calculates the mean.

# Reset the index of the pivot table and convert it to a DataFrame
pivoted = Wea_sub.pivot(columns='Temp (C)', values="Stn Press (kPa)").reset_index()

# Melt the pivoted DataFrame into a long format
print("\nMelted DataFrame:")
print(pivoted.melt(id_vars="Date/Time", value_name="My values", var_name="My var"))
# The above code converts the 'pivoted' DataFrame from wide format to long format using the 'melt' function.
# The 'Date/Time' column is retained as an identifier variable, and "Temp (C)" values are melted into "My values" column.
# The "Temp (C)" column label is used as a variable name and stored in the "My var" column.


