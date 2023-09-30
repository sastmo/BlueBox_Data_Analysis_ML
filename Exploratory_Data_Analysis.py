from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from scipy.optimize import curve_fit
from scipy import stats
import mplcursors


# Function to load and preprocess data from Excel files
def load_and_preprocess_data():
    """
    Load and preprocess data from Excel files.

    Returns:
        market_agg (DataFrame): Preprocessed data.
        unique_years (list): List of unique years in the data.
    """
    # Define file paths
    csv_file_path_MT = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\Market_Tonnes.xlsx"
    csv_file_path_CR = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\Cost_Revenue.xlsx"
    csv_file_path_FP = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\FinanceProgram.xlsx"

    # Load data into DataFrames
    market_volume = pd.read_excel(csv_file_path_MT)
    cost_revenue = pd.read_excel(csv_file_path_CR)
    payment_model = pd.read_excel(csv_file_path_FP)

    # Merge dataframes
    merge_agg = pd.merge(market_volume,
                         payment_model,
                         left_on='F key (Cost key)',
                         right_on='Foreign Key(cost key)')

    merge_agg = pd.merge(merge_agg,
                         cost_revenue,
                         left_on='F key (Cost key)',
                         right_on='Primary Key')

    # Column names for tonnage regression determined through domain knowledge and best practices
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

    # Select columns from merge_agg based on the list of column names
    market_agg = merge_agg[column_names]

    # Modify columns
    market_agg['operation cost'] = market_agg['Residential Collection Costs'] + market_agg[
        'Residential Processing Costs'] + \
                                   market_agg['Residential Depot/Transfer Costs'] + market_agg['Administration Costs']

    market_agg['Program efficiency'] = ((market_agg['Total Households Serviced'] *
                                         market_agg['TOTAL Reported and/or Calculated Marketed Tonnes']) ** 0.5) / \
                                       market_agg['operation cost']

    # Define the columns to change data types
    boolean_columns = [
        'User Pay Waste Collection (Pay-As-You-Throw)',
        'Full User Pay',
        'Partial User Pay',
        'Bag Limit Program for Garbage Collection',
        'Single Stream'
    ]

    category_columns = ['Municipal Group']

    # Change data types
    market_agg[boolean_columns] = market_agg[boolean_columns].astype(bool)
    market_agg[category_columns] = market_agg[category_columns].astype('category')

    # Create a list of unique years in the data
    unique_years = market_agg['Year'].unique()

    return market_agg, unique_years


# Function to create scatter plots for feature analysis
def create_scatter_plot(ax, x_values, y_values, x_label, y_label, program_codes, group_municipality):
    """
    Create scatter plots for feature analysis.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): Subplot to plot the scatter plot.
        x_values (array-like): X-axis values.
        y_values (array-like): Y-axis values.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        program_codes (array-like): Program codes.
        group_municipality (array-like): Municipal group labels.

    Returns:
        None
    """
    # Define a custom colormap for cluster labels
    num_group_municipality = len(set(group_municipality))
    cmap = plt.cm.get_cmap('tab10', num_group_municipality)
    colors = cycle(cmap.colors)

    # Create scatter points with different colors based on cluster labels
    scatter = sns.scatterplot(x=x_values, y=y_values, hue=group_municipality, palette=colors, ax=ax, alpha=0.7)

    # Fit a power-law curve (y = a*x^b) to the data
    '''
    def power_law(x, a, b):
        return a * np.power(x, b)

        # Fit a power-law curve (y = a*x^b) to the data
        popt, _ = curve_fit(power_law, x_values, y_values)
        a, b = popt

        # Create a smooth curve using the fitted parameters
        x_fit = np.linspace(0, 1, 100)
        y_fit = power_law(x_fit, a, b)

        sns.lineplot(x=x_fit, y=y_fit, ax=ax, color='orange', label=f'Fit (y = {a:.2f} * x^{b:.2f})')
        '''
    # Fit a linear curve to the data based on the linear regression
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)

    # Calculate the linear regression line
    y_fit = slope * x_values + intercept

    # Plot the linear regression line
    ax.plot(x_values, y_fit, color='orange', label=f'L_Reg (R²={r_value ** 2:.2f}, p={p_value:.2f}, cor={slope:.2f})')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.02, 1.2)
    ax.set_ylim(-0.02, 1.2)

    # Set aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')

    # Add a legend based on cluster labels
    ax.legend(title="Municipal Group")

    # Create hover text for scatter points
    hover_text = [f"Program Code: {code}, group_municipality: {group}" for code, group in
                  zip(program_codes, group_municipality)]

    hover = mplcursors.cursor(scatter, hover=True)
    hover.connect("add", lambda sel: sel.annotation.set_text(hover_text[sel.target.index]))


# Function to create violin plots for categorical features
def create_violin_plot(ax, x_values, y_values, x_label, y_label, hue_column):
    """
    Create violin plots for categorical features.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): Subplot to plot the violin plot.
        x_values (array-like): X-axis values.
        y_values (array-like): Y-axis values.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        hue_column (array-like): Hue column for coloring the violin plots.

    Returns:
        None
    """
    sns.violinplot(x=x_values, y=y_values, ax=ax, inner='quartile', hue=hue_column, palette='Set1')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend().set_visible(False)  # Hide the legend


# Function to create subplots and visualize features
def visualize_features(market_agg, unique_years, feature_columns, threshold=20):
    """
    Create subplots and visualize features.

    Args:
        market_agg (DataFrame): Preprocessed data.
        unique_years (list): List of unique years in the data.
        feature_columns (list): List of feature columns to visualize.
        threshold (float): Z-score threshold for outlier removal.

    Returns:
        None
    """
    # Loop through each feature
    for feature in feature_columns:
        # Create subplots for each year
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharey=True)
        plt.subplots_adjust(wspace=0.3)

        # Loop through each year
        for j, year in enumerate(unique_years):
            year_data = market_agg[market_agg['Year'] == year]

            if market_agg[feature].dtype == bool or market_agg[feature].dtype == 'category':
                create_violin_plot(axes[j], year_data[feature],
                                   year_data['TOTAL Reported and/or Calculated Marketed Tonnes'],
                                   f'{feature}', 'TOTAL Marketed Tonnes', year_data['Municipal Group'])
            else:
                # Normalize both axes
                year_data['x_values'] = (year_data[feature] - year_data[feature].min()) / (
                        year_data[feature].max() - year_data[feature].min())
                year_data['y_values'] = (year_data['TOTAL Reported and/or Calculated Marketed Tonnes'] - year_data[
                    'TOTAL Reported and/or Calculated Marketed Tonnes'].min()) / (
                                                year_data['TOTAL Reported and/or Calculated Marketed Tonnes'].max() -
                                                year_data[
                                                    'TOTAL Reported and/or Calculated Marketed Tonnes'].min())

                # Calculate Z-scores for both dimensions
                z_scores_x = stats.zscore(year_data['x_values'])
                z_scores_y = stats.zscore(year_data['y_values'])

                # Find and remove outliers based on the Z-score threshold for both dimensions
                year_data = year_data[((z_scores_x < threshold) & (z_scores_x > -threshold)) &
                                      ((z_scores_y < threshold) & (z_scores_y > -threshold))]

                create_scatter_plot(axes[j], year_data['x_values'], year_data['y_values'], f'Normalized {feature}',
                                    'Normalized TOTAL Marketed Tonnes', year_data['Program Code'],
                                    year_data['Municipal Group'])

            axes[j].set_title(f'Year {year}')
            axes[j].grid(False)

        plt.tight_layout()
        plt.suptitle(
            f'Scatter Plots: Normalized TOTAL Reported and/or Calculated Marketed Tonnes vs Normalized {feature}',
            y=1.02)

        # Remove the grid
        plt.grid(False)

        plt.show()


# Call the function to load and preprocess the data
market_agg, unique_years = load_and_preprocess_data()

# List of columns to exclude from the scatter plots (identifiers and target column)
exclude_columns = ['Year', 'Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes']

# Get the list of feature columns
feature_columns = [col for col in market_agg.columns if col not in exclude_columns]

# Set the style for the plots
sns.set(style="whitegrid")

# Calculate the number of rows and columns needed for subplots
num_rows = 1
num_cols = len(unique_years)

# Call the function to visualize features
visualize_features(market_agg, unique_years, feature_columns)

# Define the features to visualize based on 'Total Households Serviced' Feature
correlation_features = [
    'Residential Collection Costs', 'Residential Processing Costs',
    'Residential Depot/Transfer Costs', 'Residential Promotion & Education Costs',
    'Interest on Municipal  Capital', 'Administration Costs',
    'Total Gross Revenue', 'Program efficiency'
]


# Function to create subplots and visualize features
def visualize_correlation_features(market_agg, unique_years, correlation_features):
    """
    Visualizes the correlation between normalized features and 'Total Households Serviced' for each year.

    Args:
        market_agg (DataFrame): The aggregated market data.
        unique_years (list): List of unique years.
        correlation_features (list): List of features to visualize.

    Returns:
        None
    """
    # Loop through each feature
    for feature in correlation_features:
        # Create subplots for each year
        fig, axes = plt.subplots(1, len(unique_years), figsize=(15, 5), sharey=True)
        plt.subplots_adjust(wspace=0.3)

        for i, year in enumerate(unique_years):
            year_data = market_agg[market_agg['Year'] == year]

            # Normalize both axes
            x_values = (year_data[feature] - year_data[feature].min()) / (
                    year_data[feature].max() - year_data[feature].min())

            y_values = (year_data['Total Households Serviced'] - year_data['Total Households Serviced'].min()) / (
                    year_data['Total Households Serviced'].max() - year_data['Total Households Serviced'].min())

            create_scatter_plot(axes[i], x_values, y_values, f'Normalized {feature}',
                                'Normalized Total Households Serviced', year_data['Program Code'],
                                year_data['Municipal Group'])
            axes[i].set_title(f'Year {year}')
            axes[i].grid(False)

        plt.tight_layout()
        plt.suptitle(f'Scatter Plots: Normalized Total Households Serviced vs Normalized {feature}', y=1.02)

        # Remove the grid
        plt.grid(False)

        plt.show()


# Call the function to visualize correlation features
visualize_correlation_features(market_agg, unique_years, correlation_features)


# Function to visualize 'Total Households Serviced' vs Operation Cost
def visualize_households_vs_operation_cost(market_agg, unique_years):
    """
    Visualizes the correlation between 'Total Households Serviced' and 'Operation Cost' for each year.

    Args:
        market_agg (DataFrame): The aggregated market data.
        unique_years (list): List of unique years.

    Returns:
        None
    """
    # Loop through each year
    for year in unique_years:
        year_data = market_agg[market_agg['Year'] == year]

        # Normalize 'operation cost' for the current year
        normalized_operation_cost = (year_data['operation cost'] - year_data['operation cost'].min()) / (
                year_data['operation cost'].max() - year_data['operation cost'].min())

        # Normalize 'Total Households Serviced' for the current year
        normalized_total_households = (year_data['Total Households Serviced'] - year_data[
            'Total Households Serviced'].min()) / (
                                              year_data['Total Households Serviced'].max() - year_data[
                                          'Total Households Serviced'].min())

        # Create scatter plot for the current year
        plt.figure(figsize=(8, 6))
        create_scatter_plot(plt.gca(), normalized_operation_cost, normalized_total_households,
                            'Normalized Operation Cost', 'Normalized Total Households Serviced',
                            year_data['Program Code'], year_data['Municipal Group'])
        plt.title(f'Scatter Plot: Normalized Total Households Serviced vs Normalized Operation Cost - Year {year}')
        plt.tight_layout()
        # Remove the grid
        plt.grid(False)

        plt.show()


# Call the function to visualize 'Total Households Serviced' vs Operation Cost
visualize_households_vs_operation_cost(market_agg, unique_years)


# Function to visualize 'Total Households Serviced per operation cost' vs 'TOTAL Reported and/or Calculated Marketed Tonnes'
def visualize_households_vs_operation_cost_per_tonnes(market_agg, unique_years):
    """
    Visualizes the correlation between 'Total Households Serviced per operation cost' and 'TOTAL Reported and/or Calculated Marketed Tonnes' for each year.

    Args:
        market_agg (DataFrame): The aggregated market data.
        unique_years (list): List of unique years.

    Returns:
        None
    """
    for year in unique_years:
        year_data = market_agg[market_agg['Year'] == year]

        # Calculate 'Total Households Serviced per operation cost' for the current year
        year_data['Total Households Serviced per operation cost'] = (year_data['Total Households Serviced'] /
                                                                     year_data['operation cost'])

        # Normalize 'Normalized Total Households Serviced per operation cost' for the current year
        normalized_Total_Households_operation_cost = (year_data['Total Households Serviced per operation cost'] -
                                                      year_data[
                                                          'Total Households Serviced per operation cost'].min()) / (
                                                             year_data[
                                                                 'Total Households Serviced per operation cost'].max() -
                                                             year_data[
                                                                 'Total Households Serviced per operation cost'].min())

        # Normalize 'TOTAL Reported and/or Calculated Marketed Tonnes' for the current year
        y_values = (year_data['TOTAL Reported and/or Calculated Marketed Tonnes'] - year_data[
            'TOTAL Reported and/or Calculated Marketed Tonnes'].min()) / (
                           year_data['TOTAL Reported and/or Calculated Marketed Tonnes'].max() - year_data[
                       'TOTAL Reported and/or Calculated Marketed Tonnes'].min())

        # Create scatter plot for the current year
        plt.figure(figsize=(8, 6))
        create_scatter_plot(plt.gca(), normalized_Total_Households_operation_cost, y_values,
                            'Normalized Total Households Serviced per operation cost',
                            'Normalized TOTAL Reported and/or Calculated Marketed Tonnes', year_data['Program Code'],
                            year_data['Municipal Group'])
        plt.title(f'Scatter Plot: Normalized Total Households Serviced per operation cost vs '
                  f'Normalized Total Households Serviced per operation cost - Year {year}')
        plt.tight_layout()
        # Remove the grid
        plt.grid(False)

        plt.show()


# Call the function to visualize the correlation
visualize_households_vs_operation_cost_per_tonnes(market_agg, unique_years)


# Function to visualize the correlation between Total Households Serviced interaction with operation cost
# and TOTAL Reported and/or Calculated Marketed Tonnes while removing outliers
def visualize_interaction_with_operation_cost(market_agg, unique_years, threshold=3):
    """
    Visualizes the correlation between Total Households Serviced interaction with operation cost and TOTAL Reported and/or Calculated Marketed Tonnes while removing outliers for each year.

    Args:
        market_agg (DataFrame): The aggregated market data.
        unique_years (list): List of unique years.
        threshold (float): Z-score threshold for outlier removal.

    Returns:
        None
    """
    for year in unique_years:
        year_data = market_agg[market_agg['Year'] == year].copy()

        # Calculate 'Total Households Serviced per operation cost' for the current year
        year_data.loc[:, 'Total Households Serviced * operation cost'] = (year_data['Total Households Serviced'] *
                                                                          year_data['operation cost'])

        # Normalize 'Total Households Serviced * operation cost' for the current year
        year_data['normalized_Total_Households_operation_cost'] = \
            (year_data['Total Households Serviced * operation cost'] -
             year_data['Total Households Serviced * operation cost'].min()) / (
                    year_data['Total Households Serviced * operation cost'].max() -
                    year_data['Total Households Serviced * operation cost'].min())

        # Normalize 'TOTAL Reported and/or Calculated Marketed Tonnes' for the current year
        year_data['normalized_y_values'] = (year_data['TOTAL Reported and/or Calculated Marketed Tonnes'] - year_data[
            'TOTAL Reported and/or Calculated Marketed Tonnes'].min()) / (
                                                   year_data['TOTAL Reported and/or Calculated Marketed Tonnes'].max() -
                                                   year_data[
                                                       'TOTAL Reported and/or Calculated Marketed Tonnes'].min())

        # Check if there are data points for the current year
        if not year_data.empty:

            # Calculate Z-scores for both dimensions
            z_scores_x = stats.zscore(year_data['normalized_Total_Households_operation_cost'])
            z_scores_y = stats.zscore(year_data['normalized_y_values'])

            # Find and remove outliers based on the Z-score threshold for both dimensions
            year_data = year_data[((z_scores_x < threshold) & (z_scores_x > -threshold)) &
                                  ((z_scores_y < threshold) & (z_scores_y > -threshold))]

            # Check if there are data points remaining after outlier removal
            if not year_data.empty:
                plt.figure(figsize=(8, 6))
                create_scatter_plot(plt.gca(), year_data['normalized_Total_Households_operation_cost'],
                                    year_data['normalized_y_values'],
                                    'Normalized Total Households Serviced * operation cost',
                                    'Normalized TOTAL Reported and/or Calculated Marketed Tonnes',
                                    year_data['Program Code'], year_data['Municipal Group'])
                plt.title(f'Scatter Plot: Normalized Total Households Serviced * operation cost vs '
                          f'Normalized Total Households Serviced per operation cost - Year {year} (Outliers Removed)')
                plt.tight_layout()

                # Remove the grid
                plt.grid(False)
                plt.show()
            else:
                print(f"Skipping Year {year} due to mismatched array lengths.")
        else:
            print(f"No data for Year {year}.")


# Call the function to visualize the correlation while removing outliers
visualize_interaction_with_operation_cost(market_agg, unique_years, threshold=3)


# Function to visualize the correlation between Change of Total Households Serviced
# vs Change of TOTAL Reported and/or Calculated Marketed Tonnes
def visualize_change_correlation(market_agg, unique_years, threshold=0.2):
    """
    Visualizes the correlation between the change of Total Households Serviced and the change of TOTAL Reported and/or Calculated Marketed Tonnes for each year.

    Args:
        market_agg (DataFrame): The aggregated market data.
        unique_years (list): List of unique years.
        threshold (float): Threshold for categorizing data points into quadrants.

    Returns:
        quadrant_data (DataFrame): Combined results of quadrant analysis.
    """
    # Create a DataFrame to store the quadrant analysis results
    quadrant_data = []

    for year in unique_years:
        # Create a copy of the DataFrame slice for the current year
        year_data = market_agg[market_agg['Year'] == year].copy()

        # Calculate Change of Total Households Serviced
        if year == 2019:
            year_data['change of Households Serviced'] = 0
            year_data['change of Marketed Tonnes'] = 0
        else:
            # Create a copy of the DataFrame slice for the previous year
            year_data_previous = market_agg[market_agg['Year'] == year - 1].copy()

            # Calculate the change in Total Households Serviced and Reported Marketed Tonnes from the previous year
            for program_code in year_data['Program Code'].unique():
                current_year_households = year_data[year_data['Program Code'] == program_code][
                    'Total Households Serviced'].values
                prev_year_households = year_data_previous[year_data_previous['Program Code'] == program_code][
                    'Total Households Serviced'].values

                if len(current_year_households) > 0 and len(prev_year_households) > 0:
                    year_data.loc[year_data['Program Code'] == program_code, 'change of Households Serviced'] = \
                        current_year_households[0] - prev_year_households[0]
                else:
                    # Handle the case where there are no matching program codes in the previous year
                    year_data.loc[year_data['Program Code'] == program_code, 'change of Households Serviced'] = 0

                current_year_market_tonnes = year_data[year_data['Program Code'] == program_code][
                    'TOTAL Reported and/or Calculated Marketed Tonnes'].values
                prev_year_market_tonnes = year_data_previous[year_data_previous['Program Code'] == program_code][
                    'TOTAL Reported and/or Calculated Marketed Tonnes'].values

                if len(current_year_market_tonnes) > 0 and len(prev_year_market_tonnes) > 0:
                    year_data.loc[year_data['Program Code'] == program_code, 'change of Marketed Tonnes'] = \
                        current_year_market_tonnes[0] - prev_year_market_tonnes[0]
                else:
                    # Handle the case where there are no matching program codes in the previous year
                    year_data.loc[year_data['Program Code'] == program_code, 'change of Marketed Tonnes'] = 0

        # Categorize data points into quadrants based on the threshold

        # Initialize all data points as Quadrant 4 (indicating less increase in both metrics)
        year_data['Quadrant'] = 4

        # Set data points to Quadrant 1 if they have a high increase in both metrics
        year_data.loc[(year_data['change of Households Serviced'] > threshold) &
                      (year_data['change of Marketed Tonnes'] > threshold), 'Quadrant'] = 1

        # Set data points to Quadrant 2 if they have a high increase in Households Serviced but not in Marketed Tonnes
        year_data.loc[(year_data['change of Households Serviced'] > threshold) &
                      (year_data['change of Marketed Tonnes'] <= threshold), 'Quadrant'] = 2

        # Set data points to Quadrant 3 if they have a high increase in Marketed Tonnes but not in Households Serviced
        year_data.loc[(year_data['change of Households Serviced'] <= threshold) &
                      (year_data['change of Marketed Tonnes'] > threshold), 'Quadrant'] = 3

        # Append the quadrant analysis results to the DataFrame
        quadrant_data.append(year_data)

        quadrant_counts = year_data['Quadrant'].value_counts()
        print(quadrant_counts)

        # Create scatter plot with point size based on 'Program efficiency'
        fig = px.scatter(year_data, x='change of Households Serviced', y='change of Marketed Tonnes',
                         color='Municipal Group',
                         labels={'change of Households Serviced': 'Change of Households Serviced',
                                 'change of Marketed Tonnes': 'Change of Marketed Tonnes',
                                 'Municipal Group': 'Municipal Group'},
                         title=f'Scatter Plot: Change of Reported and/or Calculated Marketed Tonnes vs Change of Households Serviced - Year {year}',
                         hover_name='Program Code', hover_data={'Program Code': True},
                         size='Program efficiency',  # Set the 'Program efficiency' as the size parameter
                         size_max=30)  # Adjust the size_max value to increase marker size

        # Remove grid
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Show the interactive plot
        fig.show()

    # Combine the results from all years
    quadrant_data = pd.concat(quadrant_data)

    print(quadrant_data, quadrant_data.columns)

    return quadrant_data


# Call the function to visualize the correlation
quadrant_data = visualize_change_correlation(market_agg, unique_years, threshold=0.2)

# Define the file path
file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\quadrant_data.csv"

# Save the 'market_agg' DataFrame to the specified file path using "with" statement
with open(file_path, 'w', newline='') as file:
    quadrant_data.to_csv(file, index=False)
