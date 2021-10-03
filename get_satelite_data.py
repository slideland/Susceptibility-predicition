## Imports
import ee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image
import IPython.display as disp
import os
import requests

## Initiate Google Earth Engine
ee.Authenticate()
ee.Initialize()

## Load coordinate data for landslides
df = pd.read_csv(os.getcwd() + "/ggli2020public.csv") #Get from https://maps.nccs.nasa.gov/arcgis/apps/MapAndAppGallery/index.html?appid=574f26408683485799d02e857e5d9521

print(df.head())
print("Data loaded.")


## 
print("First event: ", np.min(pd.to_datetime(df['date'])))
print("Last event: ", np.max(pd.to_datetime(df['date'])))

# Import Global ALOS Landforms
elv = ee.Image("CSP/ERGo/1_0/Global/ALOS_landforms")

# Downaload data

scale = 1000  # scale in meters

# Iterate throught coordiantes and download satelite image
for i in range(107, np.shape(df['longitude'])[0]):
    u_lon = df['longitude'][i] 
    u_lat = df['latitude'][i]
    u_poi = ee.Geometry.Point(u_lon, u_lat)

    roi = u_poi.buffer(scale) # Reigon of interest

    response = requests.get(elv_img.getThumbURL({
    'min': 11, 'max': 42, 'dimensions': 512, 'region': roi,
    'palette': [
    '141414', '383838', '808080', 'EBEB8F', 'F7D311', 'AA0000', 'D89382',
    'DDC9C9', 'DCCDCE', '1C6330', '68AA63', 'B5C98E', 'E1F0E5', 'a975ba',
    '6f198c'
  ]}))
    file = open(os.getcwd() + "/Data/" + str(i) + ".png" , "wb")
    file.write(response.content)
    file.close()
