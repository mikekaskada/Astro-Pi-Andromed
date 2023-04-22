# Code to Calculate the ISS Speed from postional ,time and altitude data

"""
    This code calculates the ground speed of the International Space Station (ISS) using the Skyfield library.
    The code takes as input a csv file that contains the position and time data of the ISS.

    ## Steps
    1. Read the data from the csv file using the pandas library
    2. Calculate the `Distance from the Center of Earth` by adding Earth's radius (6371 km) to the
    `Altitude (in km)`
    3. Load the Skyfield's Earth model and predict the ISS ground speed using the position and time data
        of the ISS
        - For each consecutive data point, the code calculates the time difference between the two points and
            the distance between their positions. 
        - The ground speed is then calculated as the distance divided by the time difference.
    4. Append the calculated ground speeds to the data and insert a `None` value at the beginning of the
        ground_speeds list so that it has the same length as the data.
    5. Rename the column `Elevation` to `Altitude (in km)`
    6. Save the data with the ground speeds to a new csv file.
"""

"""
    pandas: A library for data manipulation and analysis. It provides data structures for efficiently
    storing large datasets and tools for working with them.

    numpy: A library for numerical computing in Python. It provides functions for working with arrays and matrices, as well as a variety of mathematical operations.

    skyfield: A library for high precision astronomy computations. It provides tools for working with positions, times, and coordinates, as well as models of solar system objects and other astronomical data. In this code, Topos, load, and utc are imported from the skyfield.api module.
"""
import pandas as pd
import numpy as np
from skyfield.api import Topos, load, utc

# Load the data from the csv file
file_path = "C:\\Users\\MK\\Desktop\\Calculate ISS speed\\milkyway\\milkyway_data.csv"
data = pd.read_csv(file_path, parse_dates=['Date/time'])

# Calculate the distance of ISS from the center of Earth by adding the Earth's radius to the Elevation
data['Distance from the Center of Earth'] = data['Elevation'] + 6371  # Earth's radius in km

# Load Skyfield's Earth model and timescale
ts = load.timescale()
eph = load('de421.bsp')
earth = eph['earth']

# Initialize the list to store the ground speeds of ISS
ground_speeds = []

# Calculate the ground speed of ISS for each time step
for i in range(len(data) - 1):
    # Get the start and end times
    start_time = ts.utc(data['Date/time'].iloc[i].to_pydatetime().replace(tzinfo=utc))
    end_time = ts.utc(data['Date/time'].iloc[i + 1].to_pydatetime().replace(tzinfo=utc))
    
    # Get the start and end positions
    """
        We are using the Topos class from the skyfield.api module to define the positions of the ISS
        at two different points in time. The positions are defined using the latitude, longitude, and
        elevation (altitude) of the ISS. The values for these parameters are obtained from the data DataFrame
        using data['Latitude'].iloc[i], data['Longitude'].iloc[i], and data['Elevation'].iloc[i], respectively.
        The same is done for the end position.
        The Topos class takes the parameters in degrees for latitude and longitude and in meters for elevation.
        However, the elevation values in the data DataFrame are in kilometers, so they are multiplied by 1000
        to convert them to meters.
    """
    start_position = Topos(latitude_degrees=data['Latitude'].iloc[i], longitude_degrees=data['Longitude'].iloc[i], elevation_m=data['Elevation'].iloc[i] * 1000)
    end_position = Topos(latitude_degrees=data['Latitude'].iloc[i + 1], longitude_degrees=data['Longitude'].iloc[i + 1], elevation_m=data['Elevation'].iloc[i + 1] * 1000)
    
    # Convert the start and end positions to Earth-centered Earth-Fixed (ECEF) coordinates
    start_position_ecef = start_position.at(start_time).position.km
    end_position_ecef = end_position.at(end_time).position.km
    
    # Calculate the distance between the start and end positions
    """
    The function 'np.linalg.norm' calculates the Euclidean norm of the difference between the two points
    in 3D space, which is equivalent to the distance between the points
    """
    distance_km = np.linalg.norm(start_position_ecef - end_position_ecef)

    # Calculate the time delta in seconds
    """ 'end_time' and 'start_time' are two instances of the Time object from the skyfield library.
    'end_time - start_time' calculates the time difference between the two times in days.
    This time difference is then multiplied by 86400 which is the number of seconds in a day.
    This calculation results in the time delta in seconds between the two time points
    """
    time_delta_s = (end_time - start_time) * 86400  # Multiply the time difference in days by the number of seconds in a day
    
    # Calculate the ground speed
    ground_speed = distance_km / time_delta_s
    ground_speeds.append(ground_speed)
    
    # Print the ground speed of each time step
    print(f"Data point {i + 1}: Ground speed: {ground_speed:.3f} km/s")

# Insert a None value at the end of the ground_speeds list so that it has the same length as the data
ground_speeds.append(None)

# Rename the column 'Elevation' to 'Altitude (in km)'
data = data.rename(columns={'Elevation': 'Altitude (in km)'})

# Add the calculated ground speed to the data
data['Speed (in km/s)'] = ground_speeds

# Convert the 'FileNo' column to integer type
data['FileNo'] = data['FileNo'].astype(int)

# Save the data with the calculated speed to a new csv file
data.to_csv("ISS data with speed.csv", index=False, header=True)

