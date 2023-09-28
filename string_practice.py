import pandas as pd

# Read the CSV file
titanic = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\titanic.csv")
titanic["surname"] = titanic["Name"].str.split(",").str.get(0)
# print(titanic)
mr = titanic["Name"].str.contains("Mrs")
# print(titanic[mr])
max_id = titanic.loc[titanic["Name"].str.len().idxmax(), "Name"]
# print(max_id)

titanic["Short_sex"] = titanic["Sex"].replace({"male": "M", "female": "F"})
print(titanic)


# Define a custom function to map 'Sex' values to 'F' (female) or 'M' (male)
def map_sex(x):
    return 'F' if x == 'female' else 'M'


# Apply the custom function to the 'Sex' column and store the result back in the 'Sex' column
titanic["Sex"] = titanic["Sex"].apply(map_sex)
print(titanic[["Sex", "Short_sex"]])