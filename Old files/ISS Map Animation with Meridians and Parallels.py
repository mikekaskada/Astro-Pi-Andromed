import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import imageio.v2 as imageio

# Load the data from the csv file
file_path = "ISS data with direction or heading and compass sign.csv"
data = pd.read_csv(file_path)

# Create the images for the first 10 time steps
images = []
for i, row in data[:10].iterrows():
    fig = plt.figure(figsize=(12, 8))
    m = Basemap(projection='cyl', resolution='l', llcrnrlat=row['Latitude'] - 10, urcrnrlat=row['Latitude'] + 10, llcrnrlon=row['Longitude'] - 10, urcrnrlon=row['Longitude'] + 10)
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.scatter(row['Longitude'], row['Latitude'], latlon=True, c='red', s=1)
    plt.title(f"Time step {i + 1}")
    plt.savefig(f"image{i + 1}.png")
    images.append(imageio.imread(f"image{i + 1}.png"))
    plt.close()
    
    # Display each image
    plt.imshow(images[i])
    plt.show()
