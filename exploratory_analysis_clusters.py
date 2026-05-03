from itertools import cycle

import mplcursors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.optimize import curve_fit
import plotly.express as px


# Load data:
# Define the file path
file_path_classification = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\result_classification.csv"

# Read the CSV file into a DataFrame
classified_data = pd.read_csv(file_path_classification)

print(classified_data.columns)


# Function to plot grouped bar charts for cluster analysis
def plot_grouped_bar_chart(data, parameters, cluster_gap=0.5):
    """
    Plots grouped bar charts for cluster analysis.

    Args:
    data (DataFrame): The dataset containing cluster information.
    parameters (list): List of parameter names to include in the chart.
    cluster_gap (float, optional): The gap between clusters. Defaults to 0.5.

    Returns:
    None
    """
    years = data['Year'].unique()

    for year in years:
        year_data = data[data['Year'] == year]
        cluster_labels = year_data['Cluster_Labels'].unique()

        plt.figure(figsize=(12, 6))
        avg_param_values_dict = {}

        for cluster in cluster_labels:
            cluster_data = year_data[year_data['Cluster_Labels'] == cluster]
            avg_param_values = [cluster_data[param].mean() for param in parameters]
            avg_param_values_dict[cluster] = avg_param_values

        # Sort the parameters based on their average values for the first cluster
        sorted_params = sorted(parameters,
                               key=lambda param: avg_param_values_dict[cluster_labels[0]][parameters.index(param)],
                               reverse=True)

        for param in sorted_params:
            avg_param_values = [avg_param_values_dict[cluster][parameters.index(param)] for cluster in cluster_labels]

            # Calculate the width of each bar for positioning
            bar_width = 0.2
            bar_positions = np.arange(len(cluster_labels))
            # Adjust the bar positions based on the parameter
            adjusted_positions = bar_positions - (
                    len(sorted_params) - sorted_params.index(param) - 1) * bar_width + cluster_gap / 2
            plt.bar(adjusted_positions, avg_param_values, width=bar_width, label=param)

        plt.title(f'Cluster Parameters for Year {year}')
        plt.xlabel('Cluster Labels')
        plt.ylabel('Average Parameter Values')

        # Use numerical cluster labels on the x-axis
        plt.xticks(bar_positions, cluster_labels)

        plt.legend(title='Parameter', loc='upper right')
        plt.tight_layout()
        plt.show()


# Years and parameters to include in the grouped bar chart
years = classified_data['Year'].unique()

parameters_1 = ['Residential Promotion & Education Costs',
                'Interest on Municipal  Capital']

plot_grouped_bar_chart(classified_data, parameters_1)

parameters_2 = ['Total Gross Revenue', 'operation cost']

plot_grouped_bar_chart(classified_data, parameters_2)

parameters_3 = ['Total Households Serviced', 'TOTAL Reported and/or Calculated Marketed Tonnes']

plot_grouped_bar_chart(classified_data, parameters_3)

# Set the style for the plots
sns.set(style="whitegrid")

# Calculate the number of rows and columns needed for subplots
num_rows = 1
num_cols = len(years)


def power_law(x, a, b):
    return a * np.power(x, b)


def create_scatter_plot(ax, x_values, y_values, x_label, y_label, program_codes, cluster_labels):
    # Define a custom colormap for cluster labels
    num_clusters = len(set(cluster_labels))
    cmap = plt.cm.get_cmap('tab10', num_clusters)
    colors = cycle(cmap.colors)

    # Create scatter points with different colors based on cluster labels
    scatter = sns.scatterplot(x=x_values, y=y_values, hue=cluster_labels, palette=colors, ax=ax, alpha=0.7)

    # Fit a power-law curve (y = a*x^b) to the data
    popt, _ = curve_fit(power_law, x_values, y_values)
    a, b = popt

    # Create a smooth curve using the fitted parameters
    x_fit = np.linspace(0, 1, 100)
    y_fit = power_law(x_fit, a, b)

    sns.lineplot(x=x_fit, y=y_fit, ax=ax, color='orange', label=f'Fit (y = {a:.2f} * x^{b:.2f})')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.02, 1.2)
    ax.set_ylim(-0.02, 1.2)

    # Set aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')

    # Add a legend based on cluster labels
    ax.legend(title="Cluster Label")

    # Create hover text for scatter points
    hover_text = [f"Program Code: {code}, Cluster: {cluster}" for code, cluster in zip(program_codes, cluster_labels)]

    hover = mplcursors.cursor(scatter, hover=True)
    hover.connect("add", lambda sel: sel.annotation.set_text(hover_text[sel.target.index]))


# Define a function for violin plot
def create_violin_plot(ax, x_values, y_values, x_label, y_label):
    sns.violinplot(x=x_values, y=y_values, ax=ax, inner='quartile')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# Function to create subplots and visualize features
def visualize_features(market_agg, unique_years, feature_columns, threshold=20):
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
                                   f'{feature}', 'TOTAL Marketed Tonnes')
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
                                    year_data['Cluster_Labels'])

            axes[j].set_title(f'Year {year}')
            axes[j].grid(False)

        plt.tight_layout()
        plt.suptitle(
            f'Scatter Plots: Normalized TOTAL Reported and/or Calculated Marketed Tonnes vs Normalized {feature}',
            y=1.02)

        # Remove the grid
        plt.grid(False)

        plt.show()


classified_data['Cluster_Labels'] = classified_data['Cluster_Labels'].astype('category')

feature_columns = ['Cluster_Labels', 'operation cost', 'Residential Promotion & Education Costs',
                   'Interest on Municipal  Capital', 'Program efficiency', 'Total Gross Revenue']

# Call the function to visualize features
visualize_features(classified_data, years, feature_columns)


# Function to plot grouped bar charts for cluster analysis based on quadrants
def plot_quadrant_counts(data, cluster_gap=0.5):
    """
    Plots grouped bar charts for cluster analysis based on quadrant counts.

    Args:
    data (DataFrame): The dataset containing cluster and quadrant information.
    cluster_gap (float, optional): The gap between clusters. Defaults to 0.5.

    Returns:
    None
    """
    years = data['Year'].unique()
    quadrants = [1, 2, 3, 4]

    for year in years:
        year_data = data[data['Year'] == year]
        cluster_labels = year_data['Cluster_Labels'].unique()

        plt.figure(figsize=(12, 6))
        quadrant_counts_dict = {}

        for cluster in cluster_labels:
            cluster_data = year_data[year_data['Cluster_Labels'] == cluster]
            quadrant_counts = [len(cluster_data[cluster_data['Quadrant'] == q]) for q in quadrants]
            quadrant_counts_dict[cluster] = quadrant_counts

        for i, q in enumerate(quadrants):
            quadrant_counts = [quadrant_counts_dict[cluster][i] for cluster in cluster_labels]

            # Calculate the width of each bar for positioning
            bar_width = 0.2
            bar_positions = np.arange(len(cluster_labels))
            # Adjust the bar positions based on the quadrant
            adjusted_positions = bar_positions - (len(quadrants) - i - 1) * bar_width + cluster_gap / 2
            plt.bar(adjusted_positions, quadrant_counts, width=bar_width, label=q)

        plt.title(f'Quadrant Counts for Year {year}')
        plt.xlabel('Cluster Labels')
        plt.ylabel('Count of Quadrants')

        # Use numerical cluster labels on the x-axis
        plt.xticks(bar_positions, cluster_labels)

        plt.legend(title='Quadrant', loc='upper right')
        plt.tight_layout()
        plt.show()


plot_quadrant_counts(classified_data)


def plot_cluster_counts_by_year(data):
    """
    Plot the count of data points in each cluster for each year.

    Parameters:
    - data: DataFrame with 'Year' and 'Cluster_Labels' columns indicating cluster membership.

    Returns:
    - None (displays the bar chart).
    """
    years = data['Year'].unique()

    for year in years:
        year_data = data[data['Year'] == year]
        cluster_counts = year_data['Cluster_Labels'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        plt.bar(cluster_counts.index, cluster_counts.values, color='green', alpha=0.8)
        plt.xlabel('Cluster Labels')
        plt.ylabel('Count')
        plt.title(f'Number of Data Points in Each Cluster for Year {year}')
        plt.xticks(cluster_counts.index)
        plt.show()


plot_cluster_counts_by_year(classified_data)


def plot_Bag_Limit_Program_counts_by_year(data):
    """
    Plot the count of data points with Bag Limit Program for Garbage Collection in each cluster for each year.

    Parameters:
    - data: DataFrame with 'Year', 'Cluster_Labels', and 'Bag Limit Program for Garbage Collection' columns.

    Returns:
    - None (displays the bar chart).
    """
    years = data['Year'].unique()

    for year in years:
        year_data = data[data['Year'] == year]
        bag_limited_counts = year_data.groupby('Cluster_Labels')[
            'Bag Limit Program for Garbage Collection'].sum().sort_index()

        plt.figure(figsize=(10, 6))
        plt.bar(bag_limited_counts.index, bag_limited_counts.values, color='skyblue')
        plt.xlabel('Cluster Labels')
        plt.ylabel('Count')
        plt.title(f'Count of Bag Limit Program for Garbage Collection in Each Cluster for Year {year}')
        plt.xticks(bag_limited_counts.index)
        plt.show()


plot_Bag_Limit_Program_counts_by_year(classified_data)

# Define the features to visualize based on 'Total Households Serviced' Feature
correlation_features = ['Residential Promotion & Education Costs', 'Total Gross Revenue', 'operation cost',
                        'Program efficiency', 'Interest on Municipal  Capital', 'Total Gross Revenue']


# Function to create subplots and visualize features
def visualize_correlation_features(market_agg, unique_years, correlation_features):
    # Loop through each feature
    for feature in correlation_features:
        # Create subplots for each year
        fig, axes = plt.subplots(1, len(unique_years), figsize=(15, 5), sharey=True)
        plt.subplots_adjust(wspace=0.3)

        for i, year in enumerate(unique_years):
            year_data = market_agg[market_agg['Year'] == year]

            # Normalize both axes
            x_values = (year_data['Total Households Serviced'] - year_data['Total Households Serviced'].min()) / (
                    year_data['Total Households Serviced'].max() - year_data['Total Households Serviced'].min())
            y_values = (year_data[feature] - year_data[feature].min()) / (
                    year_data[feature].max() - year_data[feature].min())

            create_scatter_plot(axes[i], y_values, x_values, 'Normalized Total Households Serviced',
                                f'Normalized {feature}', year_data['Program Code'], year_data['Cluster_Labels'])
            axes[i].set_title(f'Year {year}')
            axes[i].grid(False)

        plt.tight_layout()
        plt.suptitle(f'Scatter Plots: Normalized Total Households Serviced vs Normalized {feature}', y=1.02)

        # Remove the grid
        plt.grid(False)

        plt.show()


# Call the function to visualize correlation features
visualize_correlation_features(classified_data, years, correlation_features)


# Function to visualize the correlation between Change of Total Households Serviced
# vs Change of TOTAL Reported and/or Calculated Marketed Tonnes
def visualize_change_correlation(market_agg, unique_years, threshold=0.2):
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
        year_data['Cluster_Labels'] = year_data['Cluster_Labels'].astype(
            str)  # Convert to string to ensure consistent data type

        # Create scatter plot with point size based on 'Program efficiency'
        fig = px.scatter(year_data, x='change of Households Serviced', y='change of Marketed Tonnes',
                         color='Cluster_Labels',
                         labels={'change of Households Serviced': 'Change of Households Serviced',
                                 'change of Marketed Tonnes': 'Change of Marketed Tonnes',
                                 'Cluster_Labels': 'Cluster_Labels'},
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
quadrant_data = visualize_change_correlation(classified_data, years, threshold=0.2)
