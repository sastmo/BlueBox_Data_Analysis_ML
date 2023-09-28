import numpy as np
import csv
import pandas as pd
from fuzzywuzzy import fuzz


# Define the CanadaCities class
class CanadaCities:
    def __init__(self, data):
        """
        Initializes CanadaCities class with data.

        Args:
            data (DataFrame): Input data containing information about Canadian cities.
        """
        self.data = data

    columns = []

    @classmethod
    def auto_populate_columns(cls, data):
        """
        Automatically populates columns with column names and data types.

        Args:
            data (DataFrame): Input data.

        Returns:
            None
        """
        dtypes = data.dtypes
        cls.columns = list(zip(data.columns, dtypes))

    @classmethod
    def add_surrogate_key(cls, data):
        """
        Adds a surrogate key to the input data.

        Args:
            data (DataFrame): Input data.

        Returns:
            None
        """
        data['surrogate_key'] = range(1, len(data) + 1)

    @classmethod
    def read_csv(cls, file_path):
        """
        Reads data from a CSV file and initializes the CanadaCities class.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            CanadaCities: Initialized CanadaCities class with the data.
        """
        data = pd.read_csv(file_path)
        cls.auto_populate_columns(data)
        cls.add_surrogate_key(data)
        return cls(data)


# File paths
csv_file_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\canadacities.csv"
excel_file_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\Cities.xlsx"
output_data_cities_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\data_cities_output.xlsx"
output_canada_cities_path = r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\canada_cities_output.xlsx"

# Read CSV and Excel files into DataFrames
canada_cities = CanadaCities.read_csv(csv_file_path)
data_cities = pd.read_excel(excel_file_path)


# Add surrogate key to data_cities DataFrame
def add_surrogate_key(data):
    """
    Adds a surrogate key to the input DataFrame.

    Args:
        data (DataFrame): Input DataFrame.

    Returns:
        DataFrame: DataFrame with added surrogate key.
    """
    data['surrogate_key'] = range(1, len(data) + 1)
    return data


data_cities['Cities'] = data_cities['Cities'].astype(str)
data_cities = add_surrogate_key(data_cities)

# Prepare data_cities_lookup DataFrame
data_cities_lookup = data_cities.copy()
data_cities_lookup['Cities'] = data_cities['Cities'].str.lower().str.split(',')

# Prepare canada_cities_df_lookup DataFrame
canada_cities_df_lookup = canada_cities.data.copy()
canada_cities_df_lookup['city'] = canada_cities.data['city'].str.lower().str.split(',')


# Define function to generate connect_id
def generate_connect_id(row):
    """
    Generates a connect_id based on city name matching.

    Args:
        row (Series): Input row containing city information.

    Returns:
        int: Connect_id or -1 if no match is found.
    """
    city_name = row['Cities'][0]
    matching_rows = canada_cities_df_lookup[canada_cities_df_lookup['city'].apply(lambda x: x[0]) == city_name]

    if not matching_rows.empty:
        return matching_rows['surrogate_key'].iloc[0]

    best_match_score = -1
    best_match_index = -1
    for idx, city_row in canada_cities_df_lookup.iterrows():
        match_score = fuzz.partial_ratio(city_name, city_row['city'][0])
        if match_score > best_match_score:
            best_match_score = match_score
            best_match_index = idx

    if best_match_score >= 90:
        return canada_cities_df_lookup.loc[best_match_index, 'surrogate_key']

    return -1


data_cities_lookup['connect_id'] = data_cities_lookup.apply(generate_connect_id, axis=1)

# Merge DataFrames
merged_data = pd.merge(data_cities_lookup, canada_cities.data, left_on='connect_id', right_on='surrogate_key',
                       how='left')
final_data = pd.merge(data_cities, data_cities_lookup[['surrogate_key', 'connect_id']], on='surrogate_key', how='left')
final_data = final_data.merge(canada_cities.data[['surrogate_key', 'lat', 'lng']], on='surrogate_key', how='left')

# Save merged data to Excel files
final_data.to_excel(output_data_cities_path, index=False)
canada_cities.data.to_excel(output_canada_cities_path, index=False)
