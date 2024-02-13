# Occupancy Schedule Generator (OSG) for Residential Buildings
The goal of this Python package is to convert the 5-min motion detection data provided by Ecobee DYD dataset into an hourly occupancy status profile representative for the whole-house.
# How to get started?
+ Install directly from github using `pip install --upgrade git+https://github.com/AyaDoma/OccupancyScheduleGenerator-OSG-`
+ Launch Jupyter Notebook/Lab
+ `import OSG` and run the package with `OSG.start(..., df)` replacing ... with the path of the DYD raw data, and df with a dataframe of the metadata.
# OSG.start():
Once the `OSG.start()` is activated the interactive interface will initiate a sequence of operations as follows:

+ Filtering Process: The algorithms are designed to exclude any houses that do not meet the specified criteria. Specifically, it will filter out houses equipped with thermostat models lacking integrated motion sensors, as well as houses that have fewer than two operational sensors.

+ Data Aggregation: The algorithms will proceed to aggregate the data by calculating the hourly readings from the sensors.
  
+ Occupancy Status Determination: The final step involves converting the aggregated hourly sensor data into a binary occupancy status, represented by 1 (occupied) and 0 (unoccupied) considering Rule-based Algorithms.

## Rule-based Algorithms: 
After aggregating hourly reading from all sensors, the algorithms will ask for:
+ 3 distinct quantile percentages to establish thresholds for differentiating between occupied and unoccupied statuses during: working hours, non-working hours, and weekends.
+ night definition through starting and ending hours to handle the sleeping period.

These inputs will be used as follows:
+ The night definition: During the specified night hours, if any motion is detected, the entire night is deemed occupied. Conversely, if no motion is detected within these hours, the night is considered unoccupied.
+ The quantile percentages identified are used to calculate the thresholds for converting the average hourly readings into occupancy status. For any given hour during the day, if the average hourly reading exceeds the calculated threshold, the hour is deemed occupied, otherwise, the house is vacant during that hour.  




