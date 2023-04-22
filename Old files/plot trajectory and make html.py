import pandas as pd
import folium
from folium import Marker, PolyLine, Icon

# Load the data from the csv file
file_path = "ISS data with direction or heading and compass sign.csv"
data = pd.read_csv(file_path)

# Create the map
map = folium.Map(location=[0, 0], zoom_start=2)

# Add the markers for every 25th row of data to the map
for i, row in data[::25].iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Heading: {row['Direction or Heading (in degrees)']}°",
        icon=None,
        angle=row['Direction or Heading (in degrees)']
    ).add_to(map)

# Show the map
map.save('iss_trajectory_with_heading_every_25.html')
