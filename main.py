# https://pandas.pydata.org/

# Pandas can handle tabular data structure using data frame structure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

df = pd.DataFrame(
    {
        "Name": ["Morteza", "Sanaz", "Baran"],
        "Age": [32, 30, 2],
        "Sex": ["Male", "Female", "Female"]
    }
)

# print(df)
# print(df["Name"])
# relation = pd.Series(["Father", "Mother", "Daughter"], name="relations")
# print(relation)
# print(df["Age"].max())
# print(df.describe())
# print(df["Age"].sum())
# s = pd.Series(np.random.randn(1000))
# s[::2] = np.nan
# print(s.describe())
# s = pd.DataFrame(np.random.randn(1000, 5), columns=['a', 'b', 'c', 'd', 'e'])
# print(s.describe())
# print(pd.Series(["a", "b", "a", "a", "b", "b", "b"]).describe())
frame = pd.DataFrame(
    {
        "a": ["y", "y", "n", "n"],
        "b": range(4)
    }
)
# print(frame.describe(include="all"))
# print(frame.describe(include="object"))
# print(frame.describe(include="number"))

# argmin and argmax in numpy is equal to ~ idxmin and idmax respectively

# s_1 = pd.Series(np.random.randn(5))
# print(s_1)
# print(s_1.idxmin(), s_1.idxmax())
df_1 = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
# print(df_1.idxmin(axis=0))
# print(df_1.idxmax(axis=1))
# print(df_1)

# Read the CSV file
iris = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\iris.data.csv")
# Add headers to the DataFrame
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris.columns = column_names
# print(iris.head(8), iris.tail(8))
iris_ = iris.assign(sepal_ratio=iris["sepal_width"] / iris["sepal_length"])
# print(iris_)
# Calculate the sepal ratio and assign it to a new column
iris_ = iris.assign(sepal_ratio=lambda x: (x["sepal_width"] / x["sepal_length"]))
# print(iris_)

# Query over a data set

iris_ = iris.query("sepal_length > 5").assign(
    sepal_ratio=lambda x: (x["sepal_width"] / x["sepal_length"]),
    petal_ratio=lambda x: (x["petal_width"] / x["petal_length"])
)
# print(iris_)
# iris_.plot(kind="scatter", x="sepal_ratio", y="petal_ratio")
# plt.show()

# df.to_csv("Mydata.csv")
# df.to_excel("Mydata.xlsx", sheet_name="test", index=False)

dfa = pd.DataFrame(
    {
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    }
)
# dfa_ = dfa.assign(C=lambda x: x["A"] + x["B"], D=lambda x: x["A"] + x["C"])
# print(dfa_)

print(iris[5:10])
print(iris.loc[8])
print(iris.iloc[8])
print(iris["sepal_length"] > 5)
print(iris[iris["sepal_length"] > 5])

# loc vs i loc


# Create a sample DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Select the first row using iloc
first_row = df.iloc[0]
print(first_row)
# Output: A    1
#         B    4
#         Name: 0, dtype: int64

# Select the element at row 1, column 1 (zero-based indexing)
element = df.iloc[1, 1]
print(element)
# Output: 5


# Create a sample DataFrame with custom index labels
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
index_labels = ['Row1', 'Row2', 'Row3']
df = pd.DataFrame(data, index=index_labels)

# Select the row with label 'Row1' using loc
row1_data = df.loc['Row1']
print(row1_data)
# Output: A    1
#         B    4
#         Name: Row1, dtype: int64

# Select the element at row with label 'Row2', column 'B'
element = df.loc['Row2', 'B']
print(element)
# Output: 5