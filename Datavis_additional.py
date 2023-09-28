from pylab import *
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

'''plt.plot([1, 2, 3, 4])
plt.show()

plt.axis([0, 5, 0, 20])
plt.title('My first plot')
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.show()
'''

# Set the Properties of the Plot *************

'''t = np.arange(0, 2.5, 0.1)
y1_map = map(math.sin, math.pi * t)
y2_map = map(math.sin, math.pi * t + math.pi / 2)
y3_map = map(math.sin, math.pi * t - math.pi / 2)

# Convert the map objects to lists for both plotting and labeling
y1_list = list(y1_map)
y2_list = list(y2_map)
y3_list = list(y3_map)

# Plot the data
plt.plot(t, y1_list, 'b*', label='y1')
plt.plot(t, y2_list, 'g^', label='y2')
plt.plot(t, y3_list, 'ys', label='y3')'''

'''plt.plot(t, y1_list, 'b--', t, y2_list, 'g', t, y3_list, 'r-.')

plt.text(t[-1] + 0.05, y1_list[-1], 'y1', color='blue', ha='left', va='center', fontsize=10)
plt.text(t[-1] + 0.05, y2_list[-1], 'y2', color='green', ha='left', va='center', fontsize=10)
plt.text(t[-1] + 0.05, y3_list[-1], 'y3', color='red', ha='left', va='center', fontsize=10)


# Find the indices of maximum and minimum values
y1_max_index = np.argmax(y1_list)
y1_min_index = np.argmin(y1_list)

y2_max_index = np.argmax(y2_list)
y2_min_index = np.argmin(y2_list)

y3_max_index = np.argmax(y3_list)
y3_min_index = np.argmin(y3_list)

# Add labels on top of the maximum and minimum data points
for x, y in [(t[y1_max_index], y1_list[y1_max_index]), (t[y1_min_index], y1_list[y1_min_index]), (t[-1], y1_list[-1])]:
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', color='blue', fontsize=8)

for x, y in [(t[y2_max_index], y2_list[y2_max_index]), (t[y2_min_index], y2_list[y2_min_index]),  (t[-1], y2_list[-1])]:
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', color='green', fontsize=8)

for x, y in [(t[y3_max_index], y3_list[y3_max_index]), (t[y3_min_index], y3_list[y3_min_index]),  (t[-1], y3_list[-1])]:
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', color='red', fontsize=8)

# Add a zero line
plt.axhline(0, color='gray', linestyle='dashed')

# Remove upper and left borders of the Axes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlabel('t')
plt.ylabel('Value')
plt.title('Sine Functions')
plt.legend()
plt.xticks(rotation=0)

plt.show()'''

# Using the kwargs *************

'''plt.plot([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4, 2, 1, 0, 1, 2, 1, 4], linewidth=2.0)
plt.show()'''

# Working with Multiple Figures and Axes *************

# subplot() function is composed of three integers. The first number defines how many parts the figure
# is split into vertically. The second number defines how many parts the figure is divided into horizontally.
# The third issue selects which is the current subplot on which you can direct commands.

'''t = np.arange(0, 5, 0.1)
y1 = np.sin(2*np.pi*t)
y2 = np.sin(2*np.pi*t)
plt.subplot(211)
plt.plot(t, y1, 'b-.')
plt.subplot(212)
plt.plot(t, y2, 'r--')
plt.show()'''

# Adding Text
# text(x,y,s, fontdict=None, **kwargs)
'''
Table 7-1. The Possible Values for the loc Keyword
Location Code Location String
0 best
1 upper-right
2 upper-left
3 lower-right
4 lower-left
5 right
6 center-left
7 center-right
8 lower-center
9 upper-center
10 center
'''
'''
# Set the axis limits
plt.axis([0, 5, 0, 20])

# Set the plot title and text properties
plt.title('My first plot', fontsize=20, fontname='Times New Roman')
plt.xlabel('Counting', color='gray')
plt.ylabel('Square values', color='gray')

# Add text annotations at specific points
plt.text(1, 1.5, 'First')
plt.text(2, 4.5, 'Second')
plt.text(3, 9.5, 'Third')
plt.text(4, 16.5, 'Fourth')

# Add a mathematical expression using LaTeX syntax
plt.text(1.1, 12, r'$y = x^2$', fontsize=20, bbox={'facecolor': 'yellow', 'alpha': 0.1})


# plt.grid(True)   # Adding a Grid

# Plot three series of data points with different markers and colors
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro', label='First series')  # Red circles
plt.plot([1, 2, 3, 4], [0.8, 3.5, 8, 15], 'g^', label='Second series')  # Green triangles
plt.plot([1, 2, 3, 4], [0.5, 2.5, 4, 12], 'b*', label='Third series')  # Blue stars

# Add a legend at the upper-left position (loc=2)
plt.legend(loc=2)

# Save the plot as an image (PNG format)
plt.savefig('my_first_plot.png')


# Convert the matplotlib plot to Plotly format
fig = go.Figure(go.Scatter(
    x=[1, 2, 3, 4],
    y=[1, 4, 9, 16],
    mode='markers',
    marker=dict(color='red', symbol='circle'),
    name='First series'
))
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4],
    y=[0.8, 3.5, 8, 15],
    mode='markers',
    marker=dict(color='green', symbol='triangle-up'),
    name='Second series'
))
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4],
    y=[0.5, 2.5, 4, 12],
    mode='markers',
    marker=dict(color='blue', symbol='star'),
    name='Third series'
))

# Update layout properties
fig.update_layout(
    title='My first plot',
    xaxis_title='Counting',
    yaxis_title='Square values',
    title_font=dict(size=20, family='Times New Roman'),
    xaxis=dict(tickfont=dict(color='gray')),
    yaxis=dict(tickfont=dict(color='gray')),
    showlegend=True,
    legend=dict(x=0, y=1)
)

# Save the plot as an HTML file
fig.write_html('my_first_plot.html')

# Display the plot
plt.show()
'''

# Handling Date Values
# Set up locators and date formatter for the x-axis
'''months = mdates.MonthLocator()  # Locator for major ticks at month intervals
days = mdates.DayLocator()  # Locator for minor ticks at day intervals
timeFmt = mdates.DateFormatter('%Y-%m')  # Date formatter for the tick labels (year-month format)

# List of dates for the events
events = [
    datetime.date(2015, 1, 23),
    datetime.date(2015, 1, 28),
    datetime.date(2015, 2, 3),
    datetime.date(2015, 2, 21),
    datetime.date(2015, 3, 15),
    datetime.date(2015, 3, 24),
    datetime.date(2015, 4, 8),
    datetime.date(2015, 4, 24)
]

# List of corresponding readings for the events
readings = [12, 22, 25, 20, 18, 15, 17, 14]

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot the data using the provided dates and readings
plt.plot(events, readings)

# Set the title and labels for the x and y axes
plt.title('Readings Over Time', color='blue')
plt.xlabel('Date', color='gray')
plt.ylabel('Readings', color='gray')

# Set the major and minor locators and formatter for the x-axis
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(timeFmt)
ax.xaxis.set_minor_locator(days)

# Remove upper and right borders of the Axes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Rotate the x-axis tick labels to be horizontal and set their color to gray
plt.xticks(rotation=0, color='gray')
plt.yticks(color='gray')

# Display the plot
plt.show()'''


# Line Chart ****************

'''# Generate values for x, ranging from -2π to 2π, with a step of 0.01
x = np.arange(-2 * np.pi, 2 * np.pi, 0.01)

# Calculate the y values for the first function: y = sin(3x) / x
y = np.sin(3 * x) / x

# Calculate the y values for the second function: y2 = sin(2x) / x
y2 = np.sin(2 * x) / x

# Calculate the y values for the third function: y3 = sin(3x) / x
y3 = np.sin(1 * x) / x

# Plot the first function: y = sin(3x) / x with black dashed line of width 3
plt.plot(x, y, 'k--', linewidth=3)

# Plot the second function: y2 = sin(2x) / x with magenta dash-dot line
plt.plot(x, y2, 'm-.')

# Plot the third function: y3 = sin(3x) / x with custom color (hex code) and dashed line
plt.plot(x, y3, color='#87a3cc', linestyle='--')

# Add title and labels for x and y axes
plt.title('Three Functions')
plt.xlabel('x')
plt.ylabel('y')

# Display the plot
plt.show()'''

'''# Create an array of x values from -2*pi to 2*pi with a step of 0.01
x = np.arange(-2 * np.pi, 2 * np.pi, 0.01)

# Calculate y values for three different functions
y = np.sin(3 * x) / x
y2 = np.sin(2 * x) / x
y3 = np.sin(x) / x

# Plot the three functions with different colors
plt.plot(x, y, color='b', label='sin(3x)/x')
plt.plot(x, y2, color='r', label='sin(2x)/x')
plt.plot(x, y3, color='g', label='sin(x)/x')

# Set custom x and y tick positions and labels
plt.xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi],
           [r'$-2\pi$', r'$-\pi$', r'$0$', r'$+\pi$', r'$+2\pi$'])
plt.yticks([-1, 0, +1, +2, +3],
           [r'$-1$', r'$0$', r'$+1$', r'$+2$', r'$+3$'])

# Annotate the plot with a limit expression at the point (0, 1)
plt.annotate(r'$\lim_{{x\to 0}}\frac{{\sin(x)}}{{x}}= 1$', xy=[0, 1], xycoords='data',
             xytext=[30, 30], fontsize=16, textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# Get the current axes
ax = plt.gca()

# Remove the right and top spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Set the x-axis ticks to be at the bottom and y-axis ticks to be on the left
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Move the bottom and left spines to position 0
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# Add a legend to the plot
plt.legend(loc='upper right')

# Add labels to the x and y axes
plt.xlabel('x')
plt.ylabel('y')

# Add a title to the plot
plt.title('Plot of sin(3x)/x, sin(2x)/x, and sin(x)/x')

# Show the plot
plt.show()'''

# Line Charts with pandas ***********

# Data for the DataFrame

'''data = {
    'series1': [1, 3, 4, 3, 5],
    'series2': [2, 4, 5, 2, 4],
    'series3': [3, 2, 3, 1, 3]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Generate an array of x values from 0 to 4
x = np.arange(5)

# Set the axis limits for the plot
plt.axis([0, 4, 0, 7])  # Adjusted the upper x-axis limit to 4 (0 to 4).

# Plot each series using the DataFrame and x values
plt.plot(x, df['series1'], label='series1')  # Specify the label for each series.
plt.plot(x, df['series2'], label='series2')
plt.plot(x, df['series3'], label='series3')

# Add a legend to the plot, using the 'loc' parameter to change the legend position
plt.legend(loc='upper right')  # Changed the location to 'upper right' for better visibility.

# Add labels to the x and y axes
plt.xlabel('x')
plt.ylabel('y')

# Add a title to the plot
plt.title('Plot of series1, series2, and series3')

# Show the plot
plt.show()'''

# Histogram ***************

# Generate a random population array of 100 elements ranging from 0 to 99 (inclusive)
'''pop = np.random.randint(0, 100, 100)

# Display the population array
print("Population array:")
print(pop)

# Create a histogram of the population data with 20 bins
n, bins, patches = plt.hist(pop, bins=20)

# Customize the appearance of the histogram
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.title('Histogram of Random Population')
plt.grid(False)

# Show the histogram plot
plt.show()'''

# Bar Chart *************

'''index = [0, 1, 2, 3, 4]
values = [5, 7, 3, 4, 6]

# Create a bar chart with the given data
plt.bar(index, values)

# Add labels to the x and y axes
plt.xlabel('Index')
plt.ylabel('Values')

# Add a title to the plot
plt.title('Bar Chart Example')

# Show the plot
plt.show()

index = np.arange(5)
values1 = [5, 7, 3, 4, 6]

# Create a bar chart with the given data
plt.bar(index, values1)

# Set custom x-axis tick positions and labels
plt.xticks(index, ['A', 'B', 'C', 'D', 'E'])

# Add labels to the x and y axes
plt.xlabel('Categories')
plt.ylabel('Values')

# Add a title to the plot
plt.title('Bar Chart Example')

# Show the plot
plt.show()


index = np.arange(5)
values1 = [5, 7, 3, 4, 6]
std1 = [0.8, 1, 0.4, 0.9, 1.3]

# Create a bar chart with error bars and legend
plt.bar(index, values1, yerr=std1, error_kw={'ecolor': '0.1', 'capsize': 4}, alpha=0.7, label='First')

# Set custom x-axis tick positions and labels
plt.xticks(index, ['A', 'B', 'C', 'D', 'E'])

# Add labels to the x and y axes
plt.xlabel('Categories')
plt.ylabel('Values')

# Add a title to the plot
plt.title('A Bar Chart with Error Bars')

# Show the legend in the upper left corner
plt.legend(loc='upper left')

# Show the plot
plt.show()


index = np.arange(5)
values1 = [5, 7, 3, 4, 6]
std1 = [0.8, 1, 0.4, 0.9, 1.3]

# Create a horizontal bar chart with error bars and legend
plt.barh(index, values1, xerr=std1, error_kw={'ecolor': '0.1', 'capsize': 6}, alpha=0.7, label='First')

# Set custom y-axis tick positions and labels
plt.yticks(index, ['A', 'B', 'C', 'D', 'E'])

# Add labels to the x and y axes
plt.xlabel('Values')
plt.ylabel('Categories')

# Add a title to the plot
plt.title('A Horizontal Bar Chart with Error Bars')

# Show the legend at the lower right corner
plt.legend(loc='lower right')

# Show the plot
plt.show()
'''

# Multiserial Bar Chart
'''
index = np.arange(5)
values1 = [5, 7, 3, 4, 6]
values2 = [6, 6, 4, 5, 7]
values3 = [5, 6, 5, 4, 6]
bw = 0.3

# Set the axis limits for the plot
plt.axis([0, 5, 0, 8])

# Add a title to the plot with increased fontsize
plt.title('A Multiseries Bar Chart', fontsize=20)

# Create a multiseries bar chart with three sets of bars
plt.bar(index, values1, bw, color='b', label='Series 1')
plt.bar(index + bw, values2, bw, color='g', label='Series 2')
plt.bar(index + 2 * bw, values3, bw, color='r', label='Series 3')

# Set custom x-axis tick positions and labels
plt.xticks(index + 1.5 * bw, ['A', 'B', 'C', 'D', 'E'])

# Add labels to the x and y axes
plt.xlabel('Categories')
plt.ylabel('Values')

# Show the legend at the upper right corner
plt.legend(loc='right')

# Show the plot
plt.show()

index = np.arange(5)
values1 = [5, 7, 3, 4, 6]
values2 = [6, 6, 4, 5, 7]
values3 = [5, 6, 5, 4, 6]
bw = 0.3

# Set the axis limits for the plot
plt.axis([0, 8, 0, 5])

# Add a title to the plot with increased fontsize
plt.title('A Multiseries Horizontal Bar Chart', fontsize=20)

# Create a multiseries horizontal bar chart with three sets of bars
plt.barh(index, values1, bw, color='b', label='Series 1')
plt.barh(index + bw, values2, bw, color='g', label='Series 2')
plt.barh(index + 2 * bw, values3, bw, color='r', label='Series 3')

# Set custom y-axis tick positions and labels
plt.yticks(index + 0.4, ['A', 'B', 'C', 'D', 'E'])

# Add labels to the x and y axes
plt.xlabel('Values')
plt.ylabel('Categories')

# Show the legend at the upper right corner
plt.legend(loc='upper right')

# Show the plot
plt.show()

index = np.arange(5)
values1 = [5, 7, 3, 4, 6]
values2 = [6, 6, 4, 5, 7]
values3 = [5, 6, 5, 4, 6]
bw = 0.3

# Set the axis limits for the plot
plt.axis([0, 8, 0, 5])

# Add a title to the plot with increased fontsize
plt.title('A Multiseries Horizontal Bar Chart', fontsize=20)

# Create a multiseries horizontal bar chart with three sets of bars and labeled bars
plt.barh(index, values1, bw, color='b', label='Series 1')
plt.barh(index + bw, values2, bw, color='g', label='Series 2')
plt.barh(index + 2 * bw, values3, bw, color='r', label='Series 3')

# Set custom y-axis tick positions and labels
plt.yticks(index + 0.4, ['A', 'B', 'C', 'D', 'E'])

# Add labels to the x and y axes
plt.xlabel('Values')
plt.ylabel('Categories')

# Show the legend at the upper right corner
plt.legend(loc='upper right')

# Add data labels for each bar in Series 1
for i, v in enumerate(values1):
    plt.text(v + 0.1, i, str(v), color='black', fontweight='bold')

# Add data labels for each bar in Series 2
for i, v in enumerate(values2):
    plt.text(v + 0.1, i + bw, str(v), color='black', fontweight='bold')

# Add data labels for each bar in Series 3
for i, v in enumerate(values3):
    plt.text(v + 0.1, i + 2 * bw, str(v), color='black', fontweight='bold')

# Show the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

data = {
    'series1': [1, 3, 4, 3, 5],
    'series2': [2, 4, 5, 2, 4],
    'series3': [3, 2, 3, 1, 3]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Plot a bar chart for each series in the DataFrame
ax = df.plot(kind='bar')

# Add labels to the x and y axes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Add a title to the plot
plt.title('Bar Chart for Series 1, Series 2, and Series 3')

# Set the x-axis tick positions and labels
index = df.index
plt.xticks(index, ['A', 'B', 'C', 'D', 'E'], rotation=0)

# Add data labels above each bar
for i, col in enumerate(df.columns):
    for j, value in enumerate(df[col]):
        plt.text(i + j * 0.1, value + 0.1, str(value), ha='center', va='bottom')

# Show the plot
plt.show()'''

# Multiseries Stacked Bar Charts **************
'''
import matplotlib.pyplot as plt
import numpy as np

series1 = np.array([3, 4, 5, 3])
series2 = np.array([1, 2, 2, 5])
series3 = np.array([2, 3, 3, 4])
index = np.arange(4)

plt.axis([-1, 4, 0, 15])

# Plot each series separately with a bottom parameter to stack them
plt.bar(index, series1, color='r', label='Series 1')
plt.bar(index, series2, color='b', bottom=series1, label='Series 2')
plt.bar(index, series3, color='g', bottom=series1 + series2, label='Series 3')

plt.xticks(index, ['Jan15', 'Feb15', 'Mar15', 'Apr15'])

# Add labels to the x and y axes
plt.xlabel('Months')
plt.ylabel('Values')

# Add a title to the plot
plt.title('Stacked Bar Chart for Series 1, Series 2, and Series 3')

# Show the legend at the upper right corner
plt.legend(loc='upper right')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

series1 = np.array([3, 4, 5, 3])
series2 = np.array([1, 2, 2, 5])
series3 = np.array([2, 3, 3, 4])
index = np.arange(4)

plt.axis([-1, 4, 0, 15])

# Plot each series separately with a bottom parameter to stack them
plt.bar(index, series1, color='r', label='Series 1')
plt.bar(index, series2, color='b', bottom=series1, label='Series 2')
plt.bar(index, series3, color='g', bottom=series1 + series2, label='Series 3')

plt.xticks(index, ['Jan15', 'Feb15', 'Mar15', 'Apr15'])

# Add labels to the x and y axes
plt.xlabel('Months')
plt.ylabel('Values')

# Add a title to the plot
plt.title('Stacked Bar Chart for Series 1, Series 2, and Series 3')

# Show the legend at the upper right corner
plt.legend(loc='upper right')

# Add data labels for each part and section of the stacked bar chart
for i, val1 in enumerate(series1):
    plt.text(i, val1 / 2, str(val1), ha='center', va='center', color='black', fontweight='bold')

for i, val2 in enumerate(series2):
    plt.text(i, val2 / 2 + series1[i], str(val2), ha='center', va='center', color='black', fontweight='bold')

for i, val3 in enumerate(series3):
    plt.text(i, val3 / 2 + series1[i] + series2[i], str(val3), ha='center', va='center', color='black', fontweight='bold')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

index = np.arange(4)
series1 = np.array([3, 4, 5, 3])
series2 = np.array([1, 2, 2, 5])
series3 = np.array([2, 3, 3, 4])

plt.axis([0, 15, -1, 4])

# Plot each series separately with a left parameter to stack them
plt.barh(index, series1, color='r', label='Series 1')
plt.barh(index, series2, color='g', left=series1, label='Series 2')
plt.barh(index, series3, color='b', left=series1 + series2, label='Series 3')

plt.yticks(index + 0.4, ['Jan15', 'Feb15', 'Mar15', 'Apr15'])

# Add labels to the x and y axes
plt.xlabel('Values')
plt.ylabel('Months')

# Add a title to the plot
plt.title('A Multiseries Horizontal Stacked Bar Chart')

# Show the legend at the lower right corner
plt.legend(loc='lower right')

# Show the plot
plt.show()
'''

# Multi-Panel Plots ****************

import matplotlib.pyplot as plt

fig = plt.figure()

# Create the outer/main axes
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Left, Bottom, Width, Height

# Create the inner axes
inner_ax = fig.add_axes([0.6, 0.6, 0.25, 0.25])  # Left, Bottom, Width, Height

# Add content to the outer/main axes
ax.plot([1, 2, 3, 4], [5, 2, 4, 6], color='b')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Main Axes')

# Add content to the inner axes
inner_ax.plot([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.1, 0.2], color='r')
inner_ax.set_xlabel('X-axis')
inner_ax.set_ylabel('Y-axis')
inner_ax.set_title('Inner Axes')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

# Create the outer/main axes
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Left, Bottom, Width, Height

# Create the inner axes
inner_ax = fig.add_axes([0.6, 0.6, 0.25, 0.25])  # Left, Bottom, Width, Height

# Data for the main plot
x1 = np.arange(10)
y1 = np.array([1, 2, 7, 1, 5, 2, 4, 2, 3, 1])

# Data for the inner plot
x2 = np.arange(10)
y2 = np.array([1, 3, 4, 5, 4, 5, 2, 6, 4, 3])

# Plot data on the main axes
ax.plot(x1, y1, label='Main Axes')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Main Axes')
ax.legend()

# Plot data on the inner axes
inner_ax.plot(x2, y2, 'r', label='Inner Axes')
inner_ax.set_xlabel('X-axis')
inner_ax.set_ylabel('Y-axis')
inner_ax.set_title('Inner Axes')
inner_ax.legend()

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Create a 3x3 grid specification
gs = plt.GridSpec(3, 3)

# Create a figure with a size of 6x6 inches
fig = plt.figure(figsize=(6, 6))

# Add subplots to the figure using the grid specification

# Subplot at row 1, spanning first two columns
fig.add_subplot(gs[0, :2])

# Subplot at row 2, spanning first two columns
fig.add_subplot(gs[1, :2])

# Subplot at row 3, first column
fig.add_subplot(gs[2, 0])

# Subplot at rows 1 and 2, last column
fig.add_subplot(gs[:2, 2])

# Subplot at row 3, spanning from the second column to the end
fig.add_subplot(gs[2, 1:])

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Create a 3x3 grid specification
gs = plt.GridSpec(3, 3)

# Create a figure with a size of 6x6 inches
fig = plt.figure(figsize=(6, 6))

# Data for plotting
x1 = np.array([1, 3, 2, 5])
y1 = np.array([4, 3, 7, 2])
x2 = np.arange(5)
y2 = np.array([3, 2, 4, 6, 4])

# Add subplots to the figure using the grid specification

# Subplot at row 1, spanning first two columns
s1 = fig.add_subplot(gs[1, :2])
s1.plot(x1, y1, 'r')

# Subplot at row 2, spanning first two columns
s2 = fig.add_subplot(gs[0, :2])
s2.bar(x2, y2)

# Subplot at row 3, first column
s3 = fig.add_subplot(gs[2, 0])
s3.barh(x2, y2, color='g')

# Subplot at rows 1 and 2, last column
s4 = fig.add_subplot(gs[:2, 2])
s4.plot(x2, y2, 'k')

# Subplot at row 3, spanning from the second column to the end
s5 = fig.add_subplot(gs[2, 1:])
s5.plot(x1, y1, 'b^', x2, y2, 'yo')

# Display the plot
plt.show()
