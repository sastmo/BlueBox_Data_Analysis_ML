"""
This script performs analysis and visualization on a dataset containing cluster information.
It includes functions to plot grouped bar charts, quadrant counts, and more for cluster analysis.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Load data:
# Define the file path
file_path_classification = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\result_classification.csv"

# Read the CSV file into a DataFrame
classified_data = pd.read_csv(file_path_classification)


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
parameters_1 = ['Residential Promotion & Education Costs', 'Interest on Municipal  Capital']
parameters_2 = ['Total Gross Revenue', 'operation cost']
parameters_3 = ['Total Households Serviced', 'TOTAL Reported and/or Calculated Marketed Tonnes']

# Plot grouped bar charts
plot_grouped_bar_chart(classified_data, parameters_1)
plot_grouped_bar_chart(classified_data, parameters_2)
plot_grouped_bar_chart(classified_data, parameters_3)


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


# Plot quadrant counts
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


# Plot cluster counts by year
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


# Plot Bag Limit Program counts by year
plot_Bag_Limit_Program_counts_by_year(classified_data)


