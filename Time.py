# Import necessary libraries
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the CSV file 'air_quality.csv' into a DataFrame 'air_q'
air_q = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\air_quality.csv",
                    encoding="Unicode_escape", parse_dates=True)

# Rename the column "date.utc" to "datetime" and convert it to pandas datetime format
air_q = air_q.rename(columns={"date.utc": "datetime"})
air_q["datetime"] = pd.to_datetime(air_q["datetime"])

# Get the minimum and maximum datetime values in the DataFrame 'air_q'
min_t = air_q["datetime"].min()
max_t = air_q["datetime"].max()

# Extract the month from the datetime and add a new column "month" to the DataFrame 'air_q'
air_q["month"] = air_q["datetime"].dt.month

# Calculate the mean air quality value for each weekday and location and store the result in 'week_location'
week_location = air_q.groupby([air_q["datetime"].dt.weekday, "location"])["value"].mean()

# Print the time range of the data (max_t - min_t), the DataFrame 'air_q', and 'week_location'
print("Time range of the data:", max_t - min_t)
print("DataFrame 'air_q':")
print(air_q)
print("Mean air quality value for each weekday and location:")
print(week_location)

# Create a bar plot showing the mean air quality value for each hour of the day
fig, axs = plt.subplots(figsize=(12, 4))
air_q.groupby(air_q["datetime"].dt.hour)["value"].mean().plot(kind='bar', rot=0, ax=axs)
plt.xlabel("Hour of day")
plt.ylabel("$no2 (µg/m³)$")
plt.show()

# Print the DataFrame 'air_q'
print("DataFrame 'air_q':")
print(air_q)

# Pivot the 'air_q' DataFrame to create a new DataFrame 'no2' with "datetime" as index, "location" as columns, and "value" as values
no2 = air_q.pivot(index="datetime", columns="location", values="value")

# Resample the 'no2' DataFrame to get the maximum value for each month and day
no2_M_max = no2.resample("M").max()
no2_D_max = no2.resample("D").max()

# Print the DataFrames 'no2_M_max' and 'no2_D_max'
print("Maximum no2 values for each month:")
print(no2_M_max)
print("Maximum no2 values for each day:")
print(no2_D_max)

# Create a line plot showing the mean air quality value for each day
no2.resample("D").mean().plot(style="-o", figsize=(12, 4))
plt.show()