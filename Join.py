import math
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Join Operation
# Creating two DataFrames 'df1' and 'df2'
df1 = pd.DataFrame({
    'key': ['k0', 'k1', 'k2', 'k3', 'k4', 'k5'],
    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
})
df2 = pd.DataFrame({
    'key': ['k0', 'k1', 'k2'],
    'B': ["B0", "B1", "B2"]
})

# Printing the DataFrames 'df1' and 'df2'
print("DataFrame df1:")
print(df1)
print("\nDataFrame df2:")
print(df2)

# Joining 'df1' and 'df2' based on the 'key' column
joined_df = df1.join(df2.set_index("key"), on="key")
print("\nJoined DataFrame:")
print(joined_df)
# The above code joins 'df1' and 'df2' based on the 'key' column and creates a new DataFrame 'joined_df'.
# The 'set_index' method is used to set the 'key' column of 'df2' as the index before joining.

# Merge Operation
# Creating two DataFrames 'dfm1' and 'dfm2'
dfm1 = pd.DataFrame({
    'l_key': ['A', 'B', 'C', 'F'],
    'A': [1, 2, 3, 4]
})
dfm2 = pd.DataFrame({
    'r_key': ['A', 'B', 'C', 'D'],
    'B': [4, 5, 6, 7]
})

# Merging 'dfm1' and 'dfm2' based on different keys and using a left join
merged_df = dfm1.merge(dfm2, left_on='l_key', right_on='r_key', how='left')
print("\nMerged DataFrame:")
print(merged_df)
# The above code merges 'dfm1' and 'dfm2' based on 'l_key' from 'dfm1' and 'r_key' from 'dfm2'.
# It performs a left join to retain all rows from 'dfm1' and matching rows from 'dfm2'.

# Concatenation Operation
# Creating two DataFrames 'dfc1' and 'dfc2'
dfc1 = pd.DataFrame({
    'key': ['B', 'B', 'A', 'C'],
    'value1': [1, 2, 3, 4]
})
dfc2 = pd.DataFrame({
    'key': ['A', 'B', 'D'],
    'value2': [4, 5, 6]
})

# Concatenating 'dfc1' and 'dfc2' along rows and ignoring the original index
concatenated_df = pd.concat([dfc2, dfc1], ignore_index=True)
print("\nConcatenated DataFrame:")
print(concatenated_df)
# The above code concatenates 'dfc1' and 'dfc2' along rows and creates a new DataFrame 'concatenated_df'.
# The 'ignore_index' parameter is set to True, which creates a new index for the concatenated DataFrame.
