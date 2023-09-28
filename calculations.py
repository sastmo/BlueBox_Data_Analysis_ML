import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

file_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\titanic.csv"
# missing_values = ["na", "nan", "na.na", "n.a", 0]
titanic = pd.read_csv(file_path)
# print(titanic.columns)
# print(titanic.describe())
# print(titanic.shape)
# print(titanic.head())
# print(titanic.tail())
ages = titanic["Age"]
# print(ages.shape)
# print(type(ages))
ages = titanic[["Age", "Sex"]]
# print(ages.shape)
# print(type(ages))

above_35 = titanic[titanic["Age"] > 35]
# print(above_35, type(above_35))
class_23 = titanic[titanic["Pclass"].isin([2, 3])] # titanic[(titanic["Pclass"]== 2) | titanic["Pclass"]== 3]
# print(class_23, type(class_23))
age_no_na = titanic[titanic["Age"].notna()]
# print(age_no_na)
adult_name = titanic.loc[titanic["Age"] > 35, "Name"]
# print(adult_name.index)
# print(titanic.iloc[9:25, 2:6])

titanic.iloc[0:3, 1] = "Morteza"
# print(titanic)

data = {
    "Team":["A", "B", "C", "D", "E", "F", "G", "H"],
    "Rank": [1, 2, 2, 3, 1, 4, 2, 3],
    "Year": [2014, 2013, 2020, 2023, 2022, 2021, 2020, 2017],
    "Scores":[872, 728, 770, 580, 987, np.nan, 888, 745]
}
df = pd.DataFrame(data)
# df.style.hide_index()
# df.to_csv("test", index=False)
# print(df.to_string(index=False))
# print(df.groupby("Team").groups)
# print(df.groupby(["Year", "Team"]).groups)
print(df.groupby("Year").groups)

grouped = df.groupby("Year")
# print(grouped.groups)
# for name, group in grouped:
#     print(name)
#     print(group)
# print(grouped.get_group(2020)["Scores"][6])
# print(grouped["Scores"].agg(np.mean))
# print(grouped["Scores"].agg(np.size)) # size (consider nalls ) vs count (Doesn't consider nalls),
# print(grouped["Scores"].size()) # size (consider nalls ) vs count (Doesn't consider nalls),
# print(grouped["Scores"].count())
# print(grouped["Scores"].agg([np.sum, np.mean, np.std]))

# score = lambda x: (x - x.mean())/x
# score = lambda x: x * 2
# print(grouped.transform(score))
score = lambda x: x / x.mean()
print(grouped["Scores"].transform(score))
filter_ = lambda x: len(x) > 1
# print(grouped.transform(filter_))
print(grouped.filter(filter_))

