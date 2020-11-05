#!/usr/bin/env python
# coding: utf-8

# # **Rent Prices & Nearby Venues Data Analysis of Munich
# 

# Determine where is the best to leave in Munich (depending on the cost and nearby venues)
# Segmenting and Clustering Venues in Neighborhoods of Munich's
# assumption: the person like this specific kind of place and he is looking for a mid HPI
# Dataset:yy
# 
# Part 1:
# Part 2:
# Part 3:

# ## PART 1/ DATA SCRAPING and CLEANING

# ### DATASET 1 - Postal Code, Latitude and Longitude

# In[780]:


#Install required Library   - HERE INCLUDE ALL LIB THAT IS UNDER
import pandas as pd
import numpy as np
import re #Python RegEx

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # map rendering library

#geocoder
get_ipython().system('conda install -c conda-forge geopy --yes')
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[781]:


#Get the data from geonames website into Pandas DF
dfs = pd.read_html('https://www.geonames.org/postal-codes/DE/BY/bayern.html')
df=dfs[2] # index 2 to import only first table from the list
df.head()


# In[782]:


#group even and odd rows
df = df.groupby(df.index // 2).agg(lambda x: x.dropna().astype(str).str.cat(sep=' '))

#drop not necessary columns
df = df.drop(columns=['Unnamed: 0', 'Country', 'Admin1', 'Admin2', 'Admin3', 'Admin4'])

# replace the matching strings 
df = df.replace(to_replace ='MÃ¼nchen', value = 'Munich', regex = True)
df = df.replace(to_replace ='/', value = ' ', regex = True)

# extract coordinates and create new column:'coordinates'
temp_df1 = df["Place"].str.split(" ", n = 2, expand = True) 
temp_df2 = df["Code"].str.split(" ", n = 2, expand = True)
df["City"]= temp_df1[0] 
df["Latitude"]= temp_df1[1]
df["Longitude"]= temp_df1[2]
df["Postal Code"]= temp_df2[0] 
df = df[["City","Postal Code","Latitude","Longitude"]]
df


#keep only Munich
df = df[df.City == 'Munich']
df= df[['City','Postal Code','Latitude','Longitude']]
df.head()


# ### DATASET 2 - Average District Monthly Rents and Postal Code

# In[783]:


# Dataset 2 contains combined dataset temp_1 and temp_2 dowloaded from the project github repository.

#dataset1
#original source: https://www.statista.com/statistics/800552/rental-costs-in-munich-germany-by-district/

#dataset2
#original source: https://www.muenchen.de/int/en/living/postal-codes.html


# In[784]:


#Download CSV file from github repository: Average_apart_rent_Munich.csv
url_temp_1='https://raw.githubusercontent.com/matborg83/Coursera_Capstone_Munich/main/Average_apart_rent_Munich.csv'
temp_1 = pd.read_csv(url_temp_1)

temp_1 = temp_1.replace(to_replace ='Muchen', value = 'Munich', regex = True) #typo error in CSV file

# Combine 'Neuhausen' and 'Nymphenburg' and calculate the mean price (to match value 'Nymphenburg' in temp_2)
temp_1 = temp_1.replace(to_replace ='Neuhausen', value = 'Nymphenburg', regex = True)
temp_1 = temp_1.groupby(['District'],as_index=False).agg({'Monthly rent (EUR/Sq meter)': 'mean'})
temp_1= temp_1.replace(to_replace ='Nymphenburg', value = 'Neuhausen-Nymphenburg', regex = True) 

temp_1 = temp_1.round({'Monthly rent (EUR/Sq meter)': 2})

temp_1


# In[785]:


#Download CSV file from github repository: Postal_Code_Munich(3).csv
url_temp_2='https://raw.githubusercontent.com/matborg83/Coursera_Capstone_Munich/main/Postal_Code_Munich(3).csv'
temp_2= pd.read_csv(url_temp_2)

# Normalizing district values to match dataset temp_1
temp_2.loc[85:100,'district'] = 'Schwabing,Schwabing-West'
temp_2 = temp_2.replace(to_replace ='Untergiesing-Harlaching', value = 'Harlaching', regex = True)
temp_2.loc[103:114,'district'] = 'Sendling,Sendling Westpark'
temp_2 = temp_2.replace(to_replace ='Pasing-Obermenzing', value = 'Pasing', regex = True)


# 10 districts are missing in temp_1 and will be dropped in temp_2 : example: Moosach, Obergiesing
# As a result 78 Postal code out of 128 contains rent average prices
# We could have merge the value price from "Munich" to the 10 missing districts but it would be rather innacurate to do so.
# Instead beautifulsoup could be have been used to scrap specialised websites for rents in Munich and find the missing average prices.

temp_2.head()


# In[786]:


df2 = pd.merge(left=temp_1, right=temp_2, left_on='District', right_on='district')
df2= df2[['Postal Code','District','Monthly rent (EUR/Sq meter)']]
df2.head()


# ### DATASET 3 - All above combined

# In[787]:


df["Postal Code"] = df["Postal Code"].astype(str).astype(int)
df3 = pd.merge(left=df, right=df2, left_on='Postal Code', right_on='Postal Code')
df3= df3[['Postal Code','District','Monthly rent (EUR/Sq meter)','Latitude','Longitude']]

df3.head()


# In[788]:


# Concat Postal Code to create new unique Postal Code with combined Districts and average monthly rent price 
temp_3 = df3.sort_values(['Postal Code','District']).groupby('Postal Code', sort=False).District.apply('//'.join).reset_index(name='Combined Districts') 
temp_4= df3.groupby(['Postal Code'],as_index=False).agg({'Monthly rent (EUR/Sq meter)': 'mean'})
temp_5 = pd.merge(left=temp_3, right=temp_4, left_on='Postal Code', right_on='Postal Code')

df3 = pd.merge(left=temp_5, right=df, left_on='Postal Code', right_on='Postal Code')
df3 = df3.drop(['City'], axis=1)
df3.head()


# ### Dataset 4 - 10 main venues by postal code using Foursquare API

# In[789]:


#import additional library
import requests


# In[790]:


# Define Foursquare Credentials and Version
CLIENT_ID = '3WTP1JKGYST5OV1CAUS3KVOB2I2PGLPTC054P2IF5NWZJYRO' # your Foursquare ID
CLIENT_SECRET = 'HUWVSOQ1NWJEU4TCJ3AMODP0X2KMXHL2UKHSIGUBFR315N2V' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[791]:


#get the top 100 venues that are in each Postal Code within a radius of 500 meters.

def getNearbyVenues(Postal_Code, latitude, longitude, radius=500):
    
    venues_list=[]
    for Postal, lat, lng in zip(Postal_Code, latitude, longitude):
        print(Postal)
            
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
            Postal, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Postal Code', 
                  'Latitude', 
                  'Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

Postal_code_venues = getNearbyVenues(Postal_Code=df3['Postal Code'],
                                   latitude=df3['Latitude'],
                                   longitude=df3['Longitude']
                                  )


# In[792]:


print(Postal_code_venues.shape)
Postal_code_venues.head()


# In[793]:


# Venues categories encountered in each Postal Code
# one hot encoding
munich_onehot = pd.get_dummies(Postal_code_venues[['Venue Category']], prefix="", prefix_sep="")

# add Postal Code column back to dataframe
munich_onehot['Postal Code'] = Postal_code_venues['Postal Code'] 

# move neighborhood column to the first column
fixed_columns = [munich_onehot.columns[-1]] + list(munich_onehot.columns[:-1])
munich_onehot = munich_onehot[fixed_columns]

munich_onehot.shape
munich_onehot.head()


# In[794]:


#Groups the above rows by Postal Code and by taking the mean of the frequency of occurrence of each category
Postal_Code_grouped = munich_onehot.groupby('Postal Code').mean().reset_index()
Postal_Code_grouped.head()


# In[795]:


# create the final dataframe and display the top 10 venues for each neighborhood
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Postal Code']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
Postal_Code_venues_sorted = pd.DataFrame(columns=columns)
Postal_Code_venues_sorted['Postal Code'] = Postal_Code_grouped['Postal Code']

for ind in np.arange(Postal_Code_grouped.shape[0]):
    Postal_Code_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Postal_Code_grouped.iloc[ind, :], num_top_venues)

Postal_Code_venues_sorted.head()


# ## PART 2/ CLUSTERING - MACHINE LEARNING

# In[796]:


#import library for clustering
from sklearn.cluster import KMeans

There are many models for clustering out there. In this notebook, we will be presenting the model that is considered one of the simplest models amongst them. Despite its simplicity, the K-means is vastly used for clustering in many data science applications, especially useful if you need to quickly discover insights from unlabeled data. In this notebook, you will learn how to use k-Means for customer segmentation.#create seaborn to see where is k maximized then modified the k below

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc



plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 




# In[797]:


# set number of clusters
kclusters = 5

munich_grouped_clustering = Postal_Code_grouped.drop('Postal Code', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(munich_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[798]:


# add clustering labels
Postal_Code_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

df4 = df3

# merge df4 with Postal_Code_venues_sorted to add latitude/longitude for each neighborhood
df4 = df4.join(Postal_Code_venues_sorted.set_index('Postal Code'), on='Postal Code')

df4.head()


# ### Cluster 1 :Transportations, restaurants, fast foods and farmer markets

# In[799]:


df4.loc[df4['Cluster Labels'] == 0, df4.columns[[0] + list(range(5, df4.shape[1]))]]


# ### Cluster 2: Transportations, restaurants, supermarket, bakery, banks

# In[800]:


df4.loc[df4['Cluster Labels'] == 1, df4.columns[[0] + list(range(5, df4.shape[1]))]]


# ### Cluster 3: Cafes, bars and restaurants

# In[801]:


df4.loc[df4['Cluster Labels'] == 2, df4.columns[[0] + list(range(5, df4.shape[1]))]]


# ### Cluster 4: Outlier.

# In[802]:


df4.loc[df4['Cluster Labels'] == 3, df4.columns[[0] + list(range(5, df4.shape[1]))]]


# ### Cluster 5: Multiple Social Venues, Accomodations

# In[803]:


df4.loc[df4['Cluster Labels'] == 4, df4.columns[[0] + list(range(5, df4.shape[1]))]]

#examine and classify the cluster
#Let’s see how many of each class is in our data set for cluster code use below formula and a loop for??
https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/coursera/ML0101EN/ML0101EN-Clas-K-Nearest-neighbors-CustCat-py-v1.ipynb?lti=true

df['custcat'].value_counts()

# ----- to be able to explain the cluster
# ### Part 3 - DATA ANALYSING and VIZUALITATION

# In[804]:


#Obtain Munich Latitude and Longitude with API Geolocator
address = 'Munich, DE'

geolocator = Nominatim(user_agent="TR_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Munich are {}, {}.'.format(latitude, longitude))


# In[805]:


#df type
df4['Latitude']= df4['Latitude'].astype(object).astype(float)
df4['Longitude']= df4['Longitude'].astype(object).astype(float)
df4.dtypes


# In[806]:


#visualize the resulting clusters
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)


# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(df4['Latitude'], df4['Longitude'], df4['Postal Code'], df4['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
    
map_clusters


# In[ ]:


# how many unique categories can be curated from all the returned venues
print('There are {} uniques categories.'.format(len(Postal_code_venues['Venue Category'].unique())))
# how many venues were returned for each Postal Code
Postal_code_venues.groupby('Postal Code').count()


#Add 5 most pop avenue per postal code and the cluster number see from -  3. Analyze Each Neighborhood


# In[ ]:


# visualize the resulting clusters on map and add chore map for rent price


# In[ ]:


#Limitation: 
- more optimize clustering could be done by finding a better k for the k-means thanks to the the Elbow Method for optimal value of k in KMeans

