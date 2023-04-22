import pandas as pd
import numpy as np
from skyfield.api import Topos, load, utc

# 1. Read the data from the csv file
file_path = "C:\\Users\\MK\\Desktop\\Calculate ISS speed\\milkyway\\milkyway_data.csv"
data = pd.read_csv(file_path, parse_dates=['Date/time'])

# 2. Find the orbital elements of ISS orbit and the orbital inclination
# Assuming the Elevation is the altitude above the Earth's surface
data['Distance from the Center of Earth'] = data['Elevation'] + 6371  # Earth's radius in km

# Load Skyfield's Earth model
ts = load.timescale()
eph = load('de421.bsp')
earth = eph['earth']

# 3. Predict the ISS ground speed using Skyfield
ground_speeds = []

for i in range(len(data) - 1):
    start_time = ts.utc(data['Date/time'].iloc[i].to_pydatetime().replace(tzinfo=utc))
    end_time = ts.utc(data['Date/time'].iloc[i + 1].to_pydatetime().replace(tzinfo=utc))
    
    start_position = Topos(latitude_degrees=data['Latitude'].iloc[i], longitude_degrees=data['Longitude'].iloc[i], elevation_m=data['Elevation'].iloc[i] * 1000)
    end_position = Topos(latitude_degrees=data['Latitude'].iloc[i + 1], longitude_degrees=data['Longitude'].iloc[i + 1], elevation_m=data['Elevation'].iloc[i + 1] * 1000)
    
    start_position_ecef = start_position.at(start_time).position.km
    end_position_ecef = end_position.at(end_time).position.km
    
    distance_km = np.linalg.norm(start_position_ecef - end_position_ecef)
    
    time_delta_s = (end_time - start_time) * 86400  # Multiply the time difference in days by the number of seconds in a day
    ground_speed = distance_km / time_delta_s
    ground_speeds.append(ground_speed)
    
    print(f"Data point {i + 1}: Ground speed: {ground_speed:.3f} km/s")
ground_speeds.insert(0,None)

data = data.rename(columns={'Elevation': 'Altitude (in km)'})

data['Speed (in km/s)'] = ground_speeds
data['FileNo'] = data['FileNo'].astype(int)
data.to_csv("ISS data with speed.csv", index=False, header=True)

