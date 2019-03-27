# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json # library to handle JSON files
from pandas.io.json import json_normalize

from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library
import matplotlib.cm as cm
import matplotlib.colors as colors
URL = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"

res = requests.get(URL).text
soup = BeautifulSoup(res,'lxml')

table=soup.find('table', class_='wikitable sortable')
#table=soup.find(‘table’,{‘class’:’wikitable sortable’})

Postcode=[]
Borough=[]
Neighbourhood=[]

df=pd.DataFrame()
   
for row in table.find_all('tr'):
    data = row.find_all('td')
    try:
        if data[1].text=='Not assigned':
            continue
       
        Postcode.append(data[0].text)
        Borough.append(data[1].text)
        Neighbourhood.append(data[2].text)
        
    except IndexError:pass
    #print("{}|{}|{}".format(Postcode, Borough,Neighbourhood))
#Postcode = [x.encode("UTF8") for x in Postcode]
#Borough = [x.encode("UTF8") for x in Borough]
#Neighbourhood = [x.encode("UTF8") for x in Neighbourhood]
df['Postcode']=Postcode
df['Borough']=Borough
df['Neighbourhood']=Neighbourhood

df = pd.DataFrame(df, columns=['Postcode', 'Borough', 'Neighbourhood'])
df = df.groupby(['Postcode','Borough'])['Neighbourhood'].apply(','.join).reset_index()

print(df.head(10))
print(df.shape)

coordinates = pd.read_csv('C:/Users/akassenev/Desktop/Projects/Segmenting-and-Clustering-Neighborhoods-in-Toronto/Geospatial_Coordinates.csv')
coordinates.columns = ['Postcode', 'Latitude' , 'Longitude']
print(coordinates.head())

toronto_data = pd.merge(df,
                 coordinates[['Postcode', 'Latitude' , 'Longitude']],
                 on='Postcode')
print(toronto_data.head())

#filtering only Toronto

toronto_data=toronto_data[toronto_data['Borough'].str.contains("Toronto")].reset_index()
print(toronto_data.head(15))
print(toronto_data.info())

##Task 3

from geopy.geocoders import Nominatim

#address = 'Toronto'
#geolocator = Nominatim(user_agent="ny_explorer")
#location = geolocator.geocode(address)
#latitude = location.latitude
#longitude = location.longitude
#print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


#ForeSquareAPI

CLIENT_ID = 'A4WQBGU1Y31X3R3IETERHVSHWWB4U1IMCGDZRZHHK3E1CTHA' # your Foursquare ID
CLIENT_SECRET = '2M5YDVZ3AHMUM3GPG0QMDEEKZ5QBDZHK3A142ZL1WLZ0BQ0A' # your Foursquare Secret
VERSION = '20190323'

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

#Get the neighborhood's name.

#Get the neighborhood's latitude and longitude values.
neighborhood_latitude = toronto_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = toronto_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = toronto_data.loc[0, 'Borough'] # neighborhood name

latitude = toronto_data.loc[0, 'Latitude']
longitude = toronto_data.loc[0, 'Longitude']
print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))
#Now, let's get the top 100 venues that are in Toronto within a radius of 500 meters.
#First, let's create the GET request URL. Name your URL url.                                                               



LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url

results = requests.get(url).json()


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))



#Let's create a function to repeat the same process to all the neighborhoods in Toronto
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

toronto_venues = getNearbyVenues(names=toronto_data['Borough'],
                                   latitudes=toronto_data['Latitude'],
                                   longitudes=toronto_data['Longitude']
                                  )

print(toronto_venues.shape)


#toronto_venues.groupby('Neighborhood').count()

print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))

# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 



# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()
toronto_onehot.shape
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped
toronto_grouped.shape

num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('/n')
    
    
    
    
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
    
    
    num_top_venues = 10

indicators = ['st', 'nd', 'rd']



# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()



# set number of clusters
kclusters = 4

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 



# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)



toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Borough')

toronto_merged.head() # check the last columns!




# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Borough'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)

map_clusters.save('C:/Users/akassenev/Desktop/Projects/Segmenting-and-Clustering-Neighborhoods-in-Toronto/map_toronto_clusters.html')

