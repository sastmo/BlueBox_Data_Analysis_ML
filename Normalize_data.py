# https://www.statology.org/normalize-data-in-python/

# Example 1: Normalize a NumPy Array
import numpy as np

# create NumPy array
data = np.array([13, 16, 19, 22, 23, 38, 47, 56, 58, 63, 65, 70, 71])

# normalize all values in array
data_norm = (data - data.min()) / (data.max() - data.min())

# view normalized values
print(data_norm, type(data_norm))

# Example 2: Normalize All Variables in Pandas DataFrame
import pandas as pd

# create DataFrame
df = pd.DataFrame({'points': [25, 12, 15, 14, 19, 23, 25, 29],
                   'assists': [5, 7, 7, 9, 12, 9, 9, 4],
                   'rebounds': [11, 8, 10, 6, 6, 5, 9, 12]})

# normalize values in every column
df_norm = (df - df.min()) / (df.max() - df.min())

# view normalized DataFrame
print(df_norm)

# Example 3: Normalize Specific Variables in Pandas DataFrame
import pandas as pd

# create DataFrame
df = pd.DataFrame({'points': [25, 12, 15, 14, 19, 23, 25, 29],
                   'assists': [5, 7, 7, 9, 12, 9, 9, 4],
                   'rebounds': [11, 8, 10, 6, 6, 5, 9, 12]})

# define columns to normalize
x = df.iloc[:, 0:2]

# normalize values in first two columns only
df.iloc[:, 0:2] = (x - x.min()) / (x.max() - x.min())

# view normalized DataFrame
print(df)
