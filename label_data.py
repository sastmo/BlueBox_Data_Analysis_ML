import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Age': [25, 30, 22, 28, None, 35],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Other', 'Male'],
    'Education': ['High School', 'College', 'Bachelor', 'College', 'Master', None]
}

df = pd.DataFrame(data)


# Function definition
def label_data(data_):
    column_type = {}
    replace_di = {}

    for col in data_.columns:
        temp = data_[col].dropna().values
        unique = np.unique(temp)

        if data_[col].dtype == 'O':
            # Categorical
            column_type[col] = 'categorical'
            r_dict = {value: i for i, value in enumerate(unique)}
            data_[col] = data_[col].replace(r_dict)
            replace_di[col] = r_dict
        else:
            if str(data_[col].dtype).startswith('int') and len(unique) <= 15:
                # Ordinal
                column_type[col] = 'ordinal'
                r_dict = {value: i for i, value in enumerate(unique)}
                data_[col] = data_[col].replace(r_dict)
                replace_di[col] = r_dict
            else:
                column_type[col] = 'continuous'

    return replace_di, column_type


# Apply the label_data function
replace_dict, column_types = label_data(df)

# Print the results
print("Replace Dictionary:")
print(replace_dict)
print("\nColumn Types:")
print(column_types)
print("\nModified DataFrame:")
print(df)
