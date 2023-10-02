"""
Introducing a meticulously crafted machine learning pipeline designed to classify and visualize Blue Box programs.

This script is your gateway to unveiling the hidden secrets concealed within the Blue Box programs.
It's more than just code; it's your compass to navigate, classify, and visualize program data with precision and clarity.

It follows these key steps:

Clean Data, Clear Insights:
Witness the magic of data cleaning and feature engineering, the foundation of every data-driven endeavor.

Clustering Brilliance:
Experience the power of HDBSCAN clustering, illuminating patterns in the data like never before.

Visualize to Believe:
Explore the enchanting world of t-SNE visualization, where data dimensions collapse into captivating visuals.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import hdbscan
from matplotlib.lines import Line2D
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px


# Function to load data from CSV files and handle 'inf' values
def load_and_clean_data(file_path_market_agg, file_path_quadrant_data):
    """
    Loads data from CSV files and handles 'inf' values in 'Program efficiency' column.

    Args:
        file_path_market_agg (str): File path for market_agg CSV file.
        file_path_quadrant_data (str): File path for quadrant_data CSV file.

    Returns:
        Tuple: Two DataFrames - market_agg and cleaned quadrant_data.
    """
    # Read the CSV files into DataFrames
    market_agg = pd.read_csv(file_path_market_agg)
    quadrant_data = pd.read_csv(file_path_quadrant_data)

    # Create a Boolean mask to identify rows with 'inf' in 'Program efficiency' column
    inf_mask = np.isinf(quadrant_data['Program efficiency'])

    # Use the mask to drop rows with 'inf' values
    quadrant_data = quadrant_data[~inf_mask]

    return market_agg, quadrant_data


# Function to add the previous year's target as a feature
def add_previous_year_target(data_set, year_):
    """
    Adds the previous year's target as a feature to the data.

    Args:
        data_set (DataFrame): Input data.
        year_ (int): Year for which to add the previous year's target.

    Returns:
        DataFrame: Data with the 'Previous_Target' feature added.
    """
    if year_ == 2019:
        # For the initial year (2019), use the mean of the target from the same Program Code
        year_data = data_set[data_set['Year'] == year_]

        # Calculate and add the previous year's target as a feature
        previous_year_avg = data_set.groupby('Program Code')['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
        year_data['Previous_Target'] = year_data['Program Code'].map(previous_year_avg)

        return year_data
    else:
        # For subsequent years, use the previous year's target
        year_data = data_set[data_set['Year'] == year_]

        # Calculate and add the previous year's target as a feature
        previous_year = year_ - 1
        previous_feature = data_set[data_set['Year'] == previous_year][['Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes']]

        # Create a mapping of 'Program Code' to target for the previous year
        previous_mapping = previous_feature.set_index('Program Code')['TOTAL Reported and/or Calculated Marketed Tonnes'].to_dict()

        # Use the mapping to create the 'Previous_Target' column in the data set
        year_data['Previous_Target'] = year_data['Program Code'].map(previous_mapping)

        return year_data


# Function to impute missing previous target values
def impute_previous_target(data_set):
    """
    Imputes missing previous target values by taking the mean of the target for the same Program Code.

    Args:
        data_set (DataFrame): Input data with 'Previous_Target' feature.

    Returns:
        None
    """
    for index_, row_ in data_set.iterrows():
        if row_.isnull().any():
            program_code = row_['Program Code']
            previous_target = data_set[data_set['Program Code'] == program_code]['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
            data_set.loc[data_set['Program Code'] == program_code, 'Previous_Target'] = previous_target


# This function performs data preprocessing, hyperparameter tuning, and clustering using HDBSCAN.
# It returns a DataFrame associating 'Program Code' with cluster labels and probabilities.
def classification(dataset, year_):
    """
    Perform clustering on the dataset and return cluster labels and probabilities.

    Parameters:
    - dataset: DataFrame containing the data to be clustered.
    - year_: The year for which clustering is performed.

    Returns:
    - cluster_mapping: DataFrame associating 'Program Code' with cluster labels and probabilities.
    """
    # Create a DataFrame to associate 'Program Code' with cluster labels
    program_code_mapping = dataset['Program Code'].reset_index()

    # Data preprocessing (handle missing values, scaling, etc. if needed)

    # Prepare the dataset for clustering by removing irrelevant columns and addressing multicollinearity.
    # We drop columns that are not needed for clustering and handle multicollinearity by retaining only one feature
    # from a set of highly correlated columns to avoid redundancy in our data.
    revised_dataset = dataset.drop(['Year', 'Program Code', 'Municipal Group', 'Quadrant',
                                    'Program efficiency', 'Single Family Dwellings',
                                    'User Pay Waste Collection (Pay-As-You-Throw)'], axis=1)

    # Create an instance of StandardScaler
    scaler = StandardScaler()

    # Fit and transform the training data using the scaler
    revised_dataset_standard = scaler.fit_transform(revised_dataset)

    # Convert the DataFrame to a NumPy ndarray
    test_data = revised_dataset_standard

    # Create an HDBSCAN clusterer with specified parameters
    model = hdbscan.HDBSCAN()

    # Perform hyperparameter tuning for the model using cross-validation search (CV Search).
    # Define a parameter grid to search
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5],
        'min_cluster_size': [5, 10],
        'min_samples': [None, 1, 2],
        'p': [None, 1, 2]
    }

    # Define a custom scoring function using silhouette_score
    def custom_silhouette_score(estimator, X):
        cluster_labels = estimator.labels_
        try:
            # Silhouette Score is only valid when there are at least 2 clusters
            return silhouette_score(X, cluster_labels)
        except:
            return 0  # Return 0 for single-cluster cases

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(custom_silhouette_score))
    grid_search.fit(test_data)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Create an HDBSCAN clusterer with the best parameters
    clusterer = hdbscan.HDBSCAN(algorithm='best', gen_min_span_tree=True, **best_params)

    # Fit the clusterer to the test data
    cluster_labels = clusterer.fit_predict(test_data)
    cluster_probs = clusterer.probabilities_

    # Create a DataFrame to associate 'Program Code' with cluster labels and cluster probabilities
    cluster_mapping = pd.DataFrame(
        {'Program Code': program_code_mapping['Program Code'], 'Cluster_Labels': cluster_labels,
         'Cluster_Probabilities': cluster_probs})

    # Set the index of cluster_mapping to match program_code_mapping
    cluster_mapping.set_index(program_code_mapping['index'], inplace=True)

    # Print cluster labels including noise
    print(f"The detail of clustering for year {year_} is:")
    for label in np.unique(clusterer.labels_):
        if label == -1:
            print(f"Noise points: {np.sum(clusterer.labels_ == label)}")
        else:
            print(f"Cluster {label}: {np.sum(clusterer.labels_ == label)} points")

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(test_data, cluster_labels)
    print("Silhouette Score:", silhouette_avg)

    # Set up plotting parameters
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_color_codes()
    plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}

    # Plot the minimum spanning tree of the clusterer
    clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                          edge_alpha=0.6,
                                          node_size=80,
                                          edge_linewidth=2)
    plt.title("Minimum Spanning Tree")
    plt.show()

    # Plot the cluster hierarchy
    clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    plt.title("Cluster Hierarchy")
    plt.show()

    # Plot the condensed cluster tree
    clusterer.condensed_tree_.plot()
    plt.title("Condensed Cluster Tree")
    plt.show()

    # Select clusters based on persistence
    clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    plt.title("Selected Clusters Based on Persistence")
    plt.show()

    # Return the DataFrame with cluster labels and probabilities
    return cluster_mapping


# Function to create a t-SNE divergence plot and visualize divergence vs perplexity
def visualize_tsne_divergence(revised_dataset_standard, year_):
    """
    Visualizes the t-SNE divergence vs perplexity.

    Parameters:
    - revised_dataset_standard (array-like): Standardized dataset for t-SNE.
    - year_ (int): The year for which the visualization is created.

    Output:
    - Displays a matplotlib plot.
    """
    perplexity_values = np.arange(5, 110, 5)
    divergence = []

    for perplexity in perplexity_values:
        model = TSNE(n_components=2, init="pca", perplexity=perplexity)
        reduced = model.fit_transform(revised_dataset_standard)
        divergence.append(model.kl_divergence_)

    # Plot t-SNE divergence vs perplexity
    plt.figure()
    plt.plot(perplexity_values, divergence, marker='o', color='red')
    plt.title(f"t-SNE Divergence vs Perplexity for year {year_}")
    plt.xlabel("Perplexity Values")
    plt.ylabel("Divergence")
    plt.grid(True)
    plt.show()


#     Create an interactive t-SNE visualization with hover labels and save average values to an Excel file.
#     This function generates a scatter plot of data points in two-dimensional t-SNE space. Each data point is
#     color-coded based on a categorical variable, and hovering over a point reveals additional information. The
#     average values of numerical features for each category are calculated and saved to an Excel file.

def plot_tsne_with_hover_labels(X_tsne, dataset, coloring_variable, title, year_):
    """
    Plot t-SNE visualization with interactive hover labels and save average values to an Excel file.

    Args:
        X_tsne (numpy.ndarray): 2D array containing t-SNE coordinates.
        dataset (pandas.DataFrame): The dataset containing the data to be visualized.
        coloring_variable (str): The categorical variable for color-coding data points.
        title (str): The title of the plot.
        year_ (int): The year associated with the data.

    Returns:
        None
    """
    # Get unique values of the coloring variable
    unique_values = dataset[coloring_variable].unique()

    # Define a custom color palette with more colors
    custom_colors = px.colors.qualitative.Set1

    # Create an empty list to hold scatter plots
    scatter_list = []

    # Create an empty list to hold hover text
    hover_text = []

    # Loop through unique values and create scatter plots
    for i, value in enumerate(unique_values):
        value_data = X_tsne[dataset[coloring_variable] == value]
        program_efficiency = dataset[dataset[coloring_variable] == value]['Program efficiency']
        program_code = dataset[dataset[coloring_variable] == value]['Program Code']

        # Define hover text for this category
        hover_text.extend([f"Program Code: {code}<br>Efficiency: {efficiency}" for code, efficiency in
                           zip(program_code, program_efficiency)])

        # Create a scatter plot trace
        scatter = go.Scatter(
            x=value_data[:, 0],
            y=value_data[:, 1],
            mode='markers',
            marker=dict(
                size=program_efficiency * 2500,
                color=custom_colors[i % len(custom_colors)],
                opacity=0.8,
                line=dict(width=0.2, color='black')
            ),
            name=str(value),  # Convert 'value' to a string
            text=hover_text,  # Assign hover text
            hoverinfo='text+x+y'
        )
        scatter_list.append(scatter)

    # Create a layout for the plot
    layout = go.Layout(
        title=f"{title} with efficiency score based on {coloring_variable} for year {year_}",
        xaxis=dict(title="First t-SNE"),
        yaxis=dict(title="Second t-SNE"),
        legend=dict(title=f"{coloring_variable}"),
    )

    # Create a figure with the scatter plots and layout
    fig = go.Figure(data=scatter_list, layout=layout)

    # Show the interactive plot
    fig.show()

    # Calculate average values for each unique value of the coloring variable
    variable_averages = dataset.groupby(coloring_variable).mean()

    # Print the averages
    print(f"{coloring_variable} Averages:")
    print(variable_averages)

    # Create a file name based on coloring_variable and year_
    file_name = f"{coloring_variable}_{year_}_averages.xlsx"

    # Specify the file path where you want to save the Excel files
    file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\Clustering\\" + file_name

    # Save the data to an Excel file
    variable_averages.to_excel(file_path)


def main_visualization(dataset, year_):
    """
    Perform t-SNE visualization on a given dataset to visualize high-dimensional data points in a 2D space.

    Parameters:
    - dataset: DataFrame containing the dataset to be visualized.
    - year_: The year associated with the dataset.

    This function first standardizes the dataset and then applies t-SNE dimensionality reduction to visualize the data
    points in a 2D space. It also prints the KL divergence for t-SNE. Finally, it generates visualizations based on
    different attributes such as clusters, municipal groups, quadrants, and bag limit programs for garbage collection.

    Note: The 'dataset' parameter should contain columns relevant to the visualization, and some columns are dropped
    for this purpose.

    Example usage:
    visualization(my_dataset, 2023)
    """

    # Make a copy of the dataset for visualization purposes
    dataset_vis = dataset

    # Drop irrelevant columns from the DataFrame
    revised_dataset = dataset.drop(['Year', 'Program Code', 'Cluster_Probabilities',
                                    'Program efficiency', 'Single Family Dwellings',
                                    'User Pay Waste Collection (Pay-As-You-Throw)'], axis=1)

    # Call the function to visualize t-SNE divergence vs perplexity
    visualize_tsne_divergence(revised_dataset, year_)

    # Create an instance of StandardScaler to standardize the data
    scaler = StandardScaler()

    # Fit and transform the training data using the scaler
    revised_dataset_standard = scaler.fit_transform(revised_dataset)

    # Call t-SNE to reduce the data to 2D for visualization
    tsne = TSNE(n_components=2, perplexity=100, random_state=42)
    X_tsne = tsne.fit_transform(revised_dataset_standard)

    # Print the KL divergence for t-SNE
    print(tsne.kl_divergence_)

    # Visualize the data based on different attributes:

    # 1. Visualization based on clusters
    plot_tsne_with_hover_labels(X_tsne, dataset_vis, 'Cluster_Labels',
                                't-SNE visualization of waste materials collection',
                                year_)

    # 2. Visualization based on Municipal Group
    plot_tsne_with_hover_labels(X_tsne, dataset_vis, 'Municipal Group',
                                't-SNE visualization of waste materials collection',
                                year_)

    # 3. Visualization based on Quadrant
    plot_tsne_with_hover_labels(X_tsne, dataset_vis, 'Quadrant', 't-SNE visualization of waste materials collection',
                                year_)

    # 4. Visualization based on Bag Limit Program for Garbage Collection
    plot_tsne_with_hover_labels(X_tsne, dataset_vis, 'Bag Limit Program for Garbage Collection',
                                't-SNE visualization of waste materials collection',
                                year_)


# Define the file path for market_agg and quadrant_data CSV
file_path_market_agg = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\market_agg.csv"
file_path_quadrant_data = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\quadrant_data.csv"

# Call the function to load and clean the data
market_agg, quadrant_data = load_and_clean_data(file_path_market_agg, file_path_quadrant_data)

# Now, 'market_agg_data' contains the cleaned 'market_agg' data
# Get unique years from the data
unique_years = market_agg['Year'].unique()

# Define a list to store the results for each year
result_classification_list = []

# Iterate through each year
for year in unique_years:
    if year < 2022:  # Condition to limit the loop

        # **** Data Processing and Integration:
        # Add the 'Quadrant' values from 'quadrant_data' to the 'market_agg' DataFrame
        market_agg['Quadrant'] = quadrant_data.loc[
            quadrant_data['Program Code'].isin(market_agg['Program Code']), 'Quadrant']

        # Add previous year's target and operation cost as additional features
        extended_dataset = add_previous_year_target(market_agg, year)

        # Impute missing values for 'Previous_Target'
        impute_previous_target(extended_dataset)

        # *** Perform Clustering
        clustered = classification(extended_dataset, year)

        # Merge the cluster_mapping DataFrame with the extended_dataset based on 'Program Code'
        extended_dataset = pd.merge(extended_dataset,
                                    clustered[['Program Code', 'Cluster_Labels', 'Cluster_Probabilities']],
                                    on='Program Code', how='left')

        # *** Visualization using t_SNE
        main_visualization(extended_dataset, year)

        # *** Append the extended dataset to the list
        result_classification_list.append(extended_dataset)

    else:
        print("Year", year, "data processing is skipped because the new version is under construction")


# Combine the results from all years
result_classification_df = pd.concat(result_classification_list, ignore_index=True)

# Print the selected features DataFrame
print(result_classification_df, result_classification_df.columns)

# Define the file path
file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\result_classification.csv"

# Save the 'result_classification_df' DataFrame to the specified file path using "with" statement
with open(file_path, 'w', newline='') as file:
    result_classification_df.to_csv(file, index=False)
