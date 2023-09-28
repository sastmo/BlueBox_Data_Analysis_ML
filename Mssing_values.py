import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

# Read the CSV file
file_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\employees.csv"
missing_values = ["na", "nan", "na.na", "n.a", 0]
employees = pd.read_csv(file_path, na_values=missing_values)
# print(employees)

employees.dropna(axis=0, inplace=True, how="any")  # any vs all

# employees["Salary"].fillna(employees["Salary"].median(), inplace=True)
# employees["Salary"].fillna(employees["Salary"].mean(), inplace=True)
# employees["Salary"].fillna(employees["Salary"].mode(), inplace=True)
# employees["Salary"].fillna(10000, inplace=True)
# employees["Salary"].fillna(method="ffill", inplace=True)
# employees["Salary"].fillna(method="bfill", inplace=True)
# employees["Salary"].interpolate(method="linear", inplace=True)
employees["Salary"].interpolate(method="polynomial", order=4, inplace=True)
# print(employees)


