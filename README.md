# Susceptibility-prediction

Landslide susceptibility prediction for "Identifying risk with science + communities" challenge in NASA International Space Apps Challenge. We did not have time to finish the prediction part, but get_satellite_data.py can be used to download satellite images or other maps of historic landslides. train_autoencoder.py trains an autoencoder on these images. The plan was to use the method proposed in [1] and also explained in https://www.youtube.com/watch?v=2K3ScZp1dXQ&t=1275s. The method uses reconstruction probabilites of an autoencoder, trained on only one type of data in binary prediction, to determine if new data is similar to the training data. Here it was trained on elevation maps of landslide locations, but the validation and final touches are yet to be finished. If the approach shows promise it could also incorporate combinations with other types of map data.


## Getting satellite data
1. Download historic landslide location data. https://maps.nccs.nasa.gov/arcgis/apps/MapAndAppGallery/index.html?appid=574f26408683485799d02e857e5d9521
2. Sign up for Google Earth Engine. https://signup.earthengine.google.com/#!/
3. Install required packages. $pip install -r requirements.txt
4. Run get_satellite_data.py. $python get_satellite_data.py
5. Log in to Google Earth Engine when prompted. If the input box does not appear, which can happen in some systems, it is advised to copy the content to a Jupyter Notebook and run the script there instead.

If you would like to download data from other maps, more maps can be found at https://developers.google.com/earth-engine/datasets. Change "elv = ee.Image("CSP/ERGo/1_0/Global/ALOS_landforms")" to a map of your choice.

## Training the autoencoder
1. Follow the steps above to download the data and install the requirements.
2. Run train_autoencoder.py. $python train_autoencoder.py



[1] An, Jinwon, and Sungzoon Cho. "Variational autoencoder based anomaly detection using reconstruction probability." Special Lecture on IE 2.1 (2015): 1-18, http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf.
