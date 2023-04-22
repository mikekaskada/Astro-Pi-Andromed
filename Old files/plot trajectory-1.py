import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Load the data from the csv file
file_path = "ISS data with direction or heading and compass sign.csv"
data = pd.read_csv(file_path)

# Create the plot
fig = plt.figure(figsize=(12, 8))
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.scatter(data['Longitude'], data['Latitude'], latlon=True, c='red', s=1)
plt.show()
