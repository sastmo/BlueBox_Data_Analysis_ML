import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

# Scatter Plot***************************

data_scat = {
    'Unemployment_Rate': [6.1, 5.8, 5.7, 5.7, 5.8, 5.6, 5.5, 5.3, 5.2, 5.2],
    'Stock_index_price': [1500, 1520, 1525, 1523, 1515, 1540, 1545, 1560, 1555, 1565]
}
df = pd.DataFrame(data_scat, columns=["Unemployment_Rate", "Stock_index_price"])
df.plot(x="Unemployment_Rate", y="Stock_index_price", kind="scatter")
# plt.show()

# Bar Plot***************************

data_bar = {
    'Country': ["USA", "Canada", "Germany", "UK", "France"],
    'GDP_per_Country': [45000, 42000, 52000, 49000, 47000]
}

df = pd.DataFrame(data_bar, columns=["Country", "GDP_per_Country"])

# Sort the DataFrame based on GDP_per_Country in descending order (max to min)
df_sorted = df.sort_values(by="GDP_per_Country", ascending=False)

# Find the maximum and minimum GDP values
max_gdp = df_sorted['GDP_per_Country'].max()
min_gdp = df_sorted['GDP_per_Country'].min()

# Define the colors for the colormap (blue and dark gray)
colors = ['darkgray', 'blue']

# Create a custom colormap that transitions gradually from blue to dark gray
cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=len(df_sorted))

# Normalize the GDP values to be within the range [0, 1]
norm = mcolors.Normalize(vmin=min_gdp, vmax=max_gdp)

# Create the bar plot and set the colors for each bar
ax = df_sorted.plot(x="Country", y="GDP_per_Country", kind="bar", legend=False)

# Get the bars in the plot
bars = ax.containers[0]

# Apply the colormap to each bar
for bar, gdp_value in zip(bars, df_sorted['GDP_per_Country']):
    bar.set_color(cmap(norm(gdp_value)))

# Rotate the x-axis tick labels to be horizontal
plt.xticks(rotation=0)

# Display the plot
# plt.show()

# Pie Plot***************************

data_pie = {
    "Tasks": [40, 60, 70]
}

df = pd.DataFrame(data_pie, columns=["Tasks"], index=["Tasks Pending", "Tasks Ongoing", "Tasks Completed"])

# Find the maximum value in the DataFrame
max_value = df['Tasks'].max()

# Create a list of colors where the maximum value is blue and the others are gray
colors = ['gray' if value != max_value else 'blue' for value in df['Tasks']]

# Create a list of explode values, where the maximum value is separated from the pie
explode = [0.1 if value == max_value else 0 for value in df['Tasks']]

# Create the pie chart and set the colors, labels, and explode values
ax = df.plot.pie(y="Tasks", autopct='%1.1f%%', figsize=(8, 5), colors=colors, labels=None, explode=explode)

# Set the title of the pie chart
ax.set_title("Task Distribution")

# Move the legend to the top right-hand side
ax.legend(loc='upper right', labels=df.index)

# Display the plot
# plt.show()

# Stack-bar Plot***************************

data_trackbar = {
    'DATE': ["2021-01-01", "2021-01-01", "2021-01-01", "2021-02-01", "2021-02-01", "2021-02-01", "2021-03-01",
             "2021-03-01", "2021-03-01", "2021-03-01"],
    "TYPE": ['A', 'B', 'C', 'A', 'B', 'C', 'B', 'A', 'B', 'B'],
    "SALES": [1000, 200, 300, 400, 1000, 700, 200, 300, 700, 400]
}

df = pd.DataFrame(data_trackbar, columns=["DATE", "TYPE", "SALES"])
df.plot(x="DATE", y="SALES", kind="bar")
# Rotate the x-axis tick labels to be horizontal
plt.xticks(rotation=0)

df_agg = df.groupby(["DATE", "TYPE"]).sum().unstack()
df_agg.plot(kind="bar", y="SALES", stacked=True)
# Rotate the x-axis tick labels to be horizontal
plt.xticks(rotation=0)

a = df_agg.fillna(0)

# Set up the figure and axis
fig, ax = plt.subplots()

# Define colors for the bars (match the number of columns in the DataFrame)
colors = ["#FF6984", "#008080", "#A52A2A"]

# Initialize the starting position for each bar
bottom = np.zeros(len(a))

# Create the stacked bar chart
for i, col in enumerate(a.columns):
    ax.bar(a.index, a[col], bottom=bottom, color=colors[i])
    bottom += np.array(a[col])

# Add labels on top of each bar
for bar in ax.patches:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + bar.get_y() + 5,
        round(bar.get_height()),
        ha="center",
        color="black",
        weight="bold",
        size=10
    )
'''
# Create the stacked bar chart
for i, col in enumerate(a.columns):
    bars = ax.bar(a.index, a[col], bottom=bottom, color=colors[i])
    bottom += np.array(a[col])

    # Add labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # To avoid adding label to empty bars
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height,  # Adjust y-position to center the label
                f"{col}, {round(height)}",  # Use f-string to format the label
                ha="center",
                va="center",  # Vertically center the label
                color="black",
                weight="bold",
                size=10
            )
'''

# Set the title and legend
ax.set_title("My Stacked Bar Chart")
ax.legend()

# Display the plot
plt.show()
