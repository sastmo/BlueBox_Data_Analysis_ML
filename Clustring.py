import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import hdbscan
from matplotlib.lines import Line2D
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px

# from Data_Model import market_agg
# from Exploratory_Data_Analysis import quadrant_data

# Load data :
# Define the file path
file_path_market_agg = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\market_agg.csv"
file_path_quadrant_data = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\quadrant_data.csv"

# Read the CSV file into a DataFrame
market_agg = pd.read_csv(file_path_market_agg)

# Read the CSV file into a DataFrame
quadrant_data = pd.read_csv(file_path_quadrant_data)

# Create a Boolean mask to identify rows with 'inf' in 'Program efficiency' column
inf_mask = np.isinf(quadrant_data['Program efficiency'])

# Use the mask to drop rows with 'inf' values
quadrant_data = quadrant_data[~inf_mask]


# Function to add the previous year's target as a feature
def previous_feature_adder(data_set, year_):
    delete_after_use = False  # Flag to indicate whether to delete after use

    if year_ == 2019:
        # For the initial year, use the mean of the target from the same Program Code
        year_data = data_set[data_set['Year'] == year_]

        # Calculate and add the previous year's target as a feature
        pre = data_set.groupby('Program Code')['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
        year_data['Previous_Target'] = year_data['Program Code'].map(pre)

        return year_data
    else:
        # For subsequent years, use the previous year's target
        year_data = data_set[data_set['Year'] == year_]

        # Calculate and add the previous year's target as a feature
        Previous_year = year_ - 1
        Previous_feature = data_set[data_set['Year'] == Previous_year][
            ['Program Code', 'TOTAL Reported and/or Calculated Marketed Tonnes']]

        # Create a mapping of 'Program Code' to target for the previous year
        previous_mapping = Previous_feature.set_index('Program Code')[
            'TOTAL Reported and/or Calculated Marketed Tonnes'].to_dict()

        # Use the mapping to create the 'Previous_Target' column in the data set
        year_data['Previous_Target'] = year_data['Program Code'].map(previous_mapping)

        return year_data


# Function to impute previous target values
def impute_previous_target(data_set):
    for index_, row_ in data_set.iterrows():
        if row_.isnull().any():
            pc_ = row_['Program Code']
            pre_ = data_set[data_set['Program Code'] == pc_]['TOTAL Reported and/or Calculated Marketed Tonnes'].mean()
            data_set.loc[data_set['Program Code'] == pc_, 'Previous_Target'] = pre_


def k_fold_target_encoding(dataset, category_col, target_col, n_splits=5, m=2, random_state=42):
    """
    Perform K-Fold Target Encoding on a dataset.

    Parameters:
    - dataset: DataFrame, the input dataset
    - category_col: str, the column to encode
    - target_col: str, the target column
    - n_splits: int, the number of K-Folds
    - m: int, weight for overall mean
    - random_state: int, random state for reproducibility

    Returns:
    - dataset: DataFrame, the dataset with a new encoded and normalized column
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    encoded_col_name = f'{category_col}_KFold_Target_Encoding'
    dataset[encoded_col_name] = np.nan

    for train_idx, val_idx in kf.split(dataset):
        train_fold = dataset.iloc[train_idx]
        valid_fold = dataset.iloc[val_idx]
        train_weights = train_fold.groupby(category_col)[target_col].count()
        train_means = train_fold.groupby(category_col)[target_col].mean()
        overall_mean = train_fold[target_col].mean()

        for idx in valid_fold.index:  # Iterate through valid_fold indices
            category = dataset.at[idx, category_col]
            n = train_weights.get(category, 0)
            weighted_mean = (n * train_means.get(category, 0) + m * overall_mean) / (n + m)

            # Update the encoded column for the current sample
            dataset.at[idx, encoded_col_name] = weighted_mean

    dataset[encoded_col_name] = dataset.groupby(category_col)[encoded_col_name].transform('mean')

    return dataset


def classification(dataset, year_, min_cluster_size=5):
    # Create a DataFrame to associate 'Program Code' with cluster labels
    program_code_mapping = dataset['Program Code'].reset_index()

    # Data preprocessing (handle missing values, scaling, etc. if needed)

    # Drop irrelevant columns from the DataFrame
    revised_dataset = dataset.drop(['Year', 'Program Code', 'Municipal Group', 'Quadrant',
                                    'Multi-Family Dwellings', 'Program efficiency', 'Single Family Dwellings',
                                    'User Pay Waste Collection (Pay-As-You-Throw)'], axis=1)

    print(revised_dataset.columns)
    # Create an instance of StandardScaler
    scaler = StandardScaler()

    # Fit and transform the training data using the scaler
    revised_dataset_standard = scaler.fit_transform(revised_dataset)

    # Convert the DataFrame to a NumPy ndarray
    test_data = revised_dataset_standard

    # Create an HDBSCAN clustered with specified parameters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)

    # Fit the clustered to the test data
    cluster_labels = clusterer.fit_predict(test_data)
    cluster_probs = clusterer.probabilities_

    # Create a DataFrame to associate 'Program Code' with cluster labels
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

    # Print the DataFrame
    return cluster_mapping


# Function to create a t-SNE divergence plot
def plot_tsne_divergence(revised_dataset_standard, year_):
    perplexity = np.arange(5, 100, 5)
    divergence = []

    for i in perplexity:
        model = TSNE(n_components=2, init="pca", perplexity=i)
        reduced = model.fit_transform(revised_dataset_standard)
        divergence.append(model.kl_divergence_)

    plt.figure()
    plt.plot(perplexity, divergence, marker='o', color='red')
    plt.title(f"t-SNE Divergence vs Perplexity for year{year_}")
    plt.xlabel("Perplexity Values")
    plt.ylabel("Divergence")
    plt.grid(True)
    plt.show()


def plot_tsne_with_coloring(X_tsne, dataset, coloring_variable, title, year_):
    # Get unique values of the coloring variable
    unique_values = dataset[coloring_variable].unique()

    # Create a figure and a list to hold scatter plots
    plt.figure()
    scatter_list = []
    legend_labels = []

    # Define a custom color palette with more colors
    custom_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink']

    # Loop through unique values and create scatter plots
    for i, value in enumerate(unique_values):
        value_data = X_tsne[dataset[coloring_variable] == value]
        program_efficiency = dataset[dataset[coloring_variable] == value]['Program efficiency']

        # Scatter plot with color based on labels (categories)
        scatter = plt.scatter(
            value_data[:, 0], value_data[:, 1],
            c=custom_colors[i % len(custom_colors)],
            s=program_efficiency * 50000,
            alpha=1,
            edgecolors='black',
            linewidth=0.25
        )
        scatter_list.append(scatter)

        # Custom label with color based on categories
        custom_label = Line2D([0], [0], marker='o', color='w', markersize=10, label=value,
                              markerfacecolor=custom_colors[i % len(custom_colors)])
        legend_labels.append(custom_label)

    # Add legend with custom legend labels
    plt.legend(handles=legend_labels, title=coloring_variable)

    # Create a colorbar based on the points' index for color differentiation
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(unique_values)))
    sm.set_array([])
    plt.colorbar(sm, label="Efficiency")  # Change the label to "Efficiency"

    plt.title(f"{title} for year {year_}")
    plt.xlabel("First t-SNE")
    plt.ylabel("Second t-SNE")

    plt.show()

    # Calculate average values for each unique value of the coloring variable
    variable_averages = dataset.groupby(coloring_variable).mean()

    # Print the averages
    print(f"{coloring_variable} Averages:")
    print(variable_averages)

    # Create a file name based on coloring_variable and year_
    file_name = f"{coloring_variable}_{year_}_averages.xlsx"

    # Specify the file path where you want to save the Excel files
    file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\Clustring\\" + file_name

    # Save the  data to an Excel file
    variable_averages.to_excel(file_path)


def plot_tsne_with_hover_labels(X_tsne, dataset, coloring_variable, title, year_):
    # Get unique values of the coloring variable
    unique_values = dataset[coloring_variable].unique()

    # Define a custom color palette with more colors
    custom_colors = px.colors.qualitative.Set1

    # Create an empty list to hold scatter plots
    scatter_list = []
    legend_labels = []

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
        title=f"{title} with efficiency score based {coloring_variable} for year {year_}",
        xaxis=dict(title="First t-SNE"),
        yaxis=dict(title="Second t-SNE"),
        legend=dict(title=f"{coloring_variable}"),
    )

    # Create a figure with the scatter plots and layout
    fig = go.Figure(data=scatter_list, layout=layout)

    # Add legend with custom legend labels
    for label in legend_labels:
        fig.add_trace(label)

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
    file_path = r"C:\Users\admin\OneDrive\Desktop\Dataset\ML\Clustring\\" + file_name

    # Save the  data to an Excel file
    variable_averages.to_excel(file_path)


def visualization(dataset, year_):
    dataset_vis = dataset
    # Drop irrelevant columns from the DataFrame
    revised_dataset = dataset.drop(['Year', 'Program Code', 'Cluster_Probabilities',
                                    'Multi-Family Dwellings', 'Program efficiency', 'Single Family Dwellings',
                                    'User Pay Waste Collection (Pay-As-You-Throw)'], axis=1)

    # Create an instance of StandardScaler
    scaler = StandardScaler()

    # Fit and transform the training data using the scaler
    revised_dataset_standard = scaler.fit_transform(revised_dataset)

    # Call the t-SNE divergence plot function
    # plot_tsne_divergence(revised_dataset_standard, year_)

    # t-SNE visualization for Customer Churn dataset
    tsne = TSNE(n_components=2, perplexity=100, random_state=42)
    X_tsne = tsne.fit_transform(revised_dataset_standard)

    # Print the KL divergence for t-SNE
    print(tsne.kl_divergence_)

    # 1.Visualization based on clusters
    plot_tsne_with_hover_labels(X_tsne, dataset_vis, 'Cluster_Labels',
                                't-SNE visualization of waste materials collection',
                                year_)

    # 2.Visualization based on Municipal Group
    plot_tsne_with_hover_labels(X_tsne, dataset_vis, 'Municipal Group',
                                't-SNE visualization of waste materials collection',
                                year_)

    # 3.Visualization based on Quadrant
    plot_tsne_with_hover_labels(X_tsne, dataset_vis, 'Quadrant', 't-SNE visualization of waste materials collection',
                                year_)

    # 4.Visualization based on Bag Limit Program for Garbage Collection
    plot_tsne_with_hover_labels(X_tsne, dataset_vis, 'Bag Limit Program for Garbage Collection', 't-SNE visualization of waste materials collection',
                                year_)


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
        extended_dataset = previous_feature_adder(market_agg, year)

        # Impute missing values for 'Previous_Target'
        impute_previous_target(extended_dataset)

        # Encoding the data
        ''' 
        # Encoding Municipal Group column
        dataset_encoded_MGroup = k_fold_target_encoding(extended_dataset, 'Municipal Group',
                                                        'TOTAL Reported and/or Calculated Marketed Tonnes')

        # Encoding  Quadrant column

        dataset_encoded = k_fold_target_encoding(extended_dataset, 'Quadrant',
                                                 'TOTAL Reported and/or Calculated Marketed Tonnes')
        
        # Encoding  Quadrant column using one-hot encoding
        extended_dataset = extended_dataset.copy()
        encoded_dataset = pd.get_dummies(extended_dataset, columns=['Quadrant'], prefix=['Quadrant'])'''

        # *** Perform Clustering
        clustered = classification(extended_dataset, year)

        print(extended_dataset.columns)

        # Merge the cluster_mapping DataFrame with the extended_dataset based on 'Program Code'
        extended_dataset = pd.merge(extended_dataset,
                                    clustered[['Program Code', 'Cluster_Labels', 'Cluster_Probabilities']],
                                    on='Program Code', how='left')

        # *** Visualization using t_SNE
        visualization(extended_dataset, year)

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

"""
The detail of clustering for year 2019 is:
Noise points: 14
Cluster 0: 15 points  
Cluster 1: 27 points
Cluster 2: 27 points
Cluster 3: 45 points
Cluster 4: 89 points
Cluster 5: 20 points
Cluster 6: 16 points
Silhouette Score: 0.8020905009425441

The detail of clustering for year 2020 is:
Noise points: 10
Cluster 0: 5 points
Cluster 1: 12 points
Cluster 2: 21 points
Cluster 3: 25 points
Cluster 4: 15 points
Cluster 5: 21 points
Cluster 6: 91 points
Cluster 7: 50 points
Silhouette Score: 0.8013274052514461

The detail of clustering for year 2021 is:
Noise points: 15
Cluster 0: 15 points
Cluster 1: 17 points
Cluster 2: 30 points
Cluster 3: 15 points
Cluster 4: 22 points
Cluster 5: 51 points
Cluster 6: 81 points
Silhouette Score: 0.788900596115757

"""
