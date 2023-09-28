import numpy as np
import pandas as pd
from pmdarima.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_data():
    """
    Load data from Excel files into DataFrames.

    Returns:
    - market_volume: DataFrame, market volume data
    - cost_revenue: DataFrame, cost and revenue data
    - payment_model: DataFrame, payment model data
    """
    # Define file paths
    csv_file_path_MT = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\Market_Tonnes.xlsx"
    csv_file_path_CR = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\Cost_Revenue.xlsx"
    csv_file_path_FP = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\FinanceProgram.xlsx"

    # Load data into DataFrames
    market_volume = pd.read_excel(csv_file_path_MT)
    cost_revenue = pd.read_excel(csv_file_path_CR)
    payment_model = pd.read_excel(csv_file_path_FP)

    return market_volume, cost_revenue, payment_model


def merge_data(market_volume, cost_revenue, payment_model):
    """
    Merge dataframes.

    Parameters:
    - market_volume: DataFrame, market volume data
    - cost_revenue: DataFrame, cost and revenue data
    - payment_model: DataFrame, payment model data

    Returns:
    - merge_agg: DataFrame, merged data
    """
    # Merge dataframes
    merge_agg = pd.merge(market_volume,
                         payment_model,
                         left_on='F key (Cost key)',
                         right_on='Foreign Key(cost key)')

    merge_agg = pd.merge(merge_agg,
                         cost_revenue,
                         left_on='F key (Cost key)',
                         right_on='Primary Key')

    return merge_agg


def preprocess_data(merge_agg):
    """
    Preprocess and clean the merged data.

    Parameters:
    - merge_agg: DataFrame, merged data

    Returns:
    - market_agg: DataFrame, preprocessed data
    """
    # Define column names for tonnage regression
    column_names = [
        'Total Households Serviced',
        'Single Family Dwellings', 'Multi-Family Dwellings',
        'User Pay Waste Collection (Pay-As-You-Throw)', 'Full User Pay',
        'Partial User Pay', 'Bag Limit Program for Garbage Collection',
        'Program Code', 'Municipal Group', 'Single Stream',
        'Residential Collection Costs', 'Residential Processing Costs',
        'Residential Depot/Transfer Costs', 'Residential Promotion & Education Costs',
        'Interest on Municipal  Capital', 'Administration Costs',
        'Total Gross Revenue', 'Year', 'TOTAL Reported and/or Calculated Marketed Tonnes']

    # Select relevant columns
    market_agg = merge_agg[column_names]

    # Calculate VIF
    def calculate_vif(data_frame):
        vif_data_ = pd.DataFrame()
        vif_data_["Variable"] = data_frame.columns
        vif_data_["VIF"] = [variance_inflation_factor(data_frame.values, i) for i in range(data_frame.shape[1])]
        return vif_data_

    # Remove Year column
    market_agg_filtered = market_agg.drop(columns=['Year'])

    # Handle missing values and calculate VIF
    numerical_data = market_agg_filtered.select_dtypes(include=[np.number])
    numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)
    numerical_data.dropna(inplace=True)
    vif_data_1 = calculate_vif(numerical_data)

    # Modify columns
    market_agg['operation cost'] = market_agg['Residential Collection Costs'] + market_agg[
        'Residential Processing Costs'] + \
                                   market_agg['Residential Depot/Transfer Costs'] + market_agg['Administration Costs']

    market_agg['Program efficiency'] = ((market_agg['Total Households Serviced'] *
                                         market_agg['TOTAL Reported and/or Calculated Marketed Tonnes']) ** 0.5) / \
                                       market_agg['operation cost']

    market_agg['Interaction of Households Serviced and operation cost'] = market_agg['operation cost'] * market_agg[
        'Total Households Serviced']


    columns_to_drop = [
        'Residential Collection Costs',
        'Residential Processing Costs',
        'Residential Depot/Transfer Costs',
        'Administration Costs',
        'Partial User Pay'
    ]

    market_agg_filtered = market_agg.drop(columns=columns_to_drop)

    # Calculate VIF
    vif_data_2 = calculate_vif(market_agg_filtered)

    # Swap column positions
    column_names = market_agg_filtered.columns.tolist()
    column_index_1 = column_names.index('Interaction of Households Serviced and operation cost')
    column_index_2 = column_names.index('TOTAL Reported and/or Calculated Marketed Tonnes')
    column_names[column_index_1], column_names[column_index_2] = column_names[column_index_2], column_names[
        column_index_1]
    market_agg = market_agg_filtered[column_names]

    # Convert columns to boolean
    columns_to_convert_to_boolean = [
        'User Pay Waste Collection (Pay-As-You-Throw)',
        'Full User Pay',
        'Bag Limit Program for Garbage Collection',
        'Single Stream'
    ]

    market_agg[columns_to_convert_to_boolean] = market_agg[columns_to_convert_to_boolean].astype(bool)

    return market_agg, vif_data_1, vif_data_2


market_volume, cost_revenue, payment_model = load_data()
merge_agg = merge_data(market_volume, cost_revenue, payment_model)
market_agg, vif_data_1, vif_data_2 = preprocess_data(merge_agg)

print(market_agg.columns)
print(vif_data_1)
print(vif_data_2)

# Define the file path
file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\market_agg.csv"

# Save the 'market_agg' DataFrame to the specified file path using "with" statement
with open(file_path, 'w', newline='') as file:
    market_agg.to_csv(file, index=False)
