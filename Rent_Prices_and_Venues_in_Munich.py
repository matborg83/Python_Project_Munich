#!/usr/bin/env python
# coding: utf-8

# # Rent Prices & Nearby Venues Data Analysis of Munich
# 

# In[970]:


#Required Libraries for the project

import matplotlib.pyplot as plt
import requests
from sklearn.cluster import KMeans #library for clustering


# ## PART 1/ DATA SCRAPING and CLEANING

# ### DATASET 1 (df) - Postal Code, Latitude and Longitude

# In[971]:


#Get the data from geonames website into Pandas DF
dfs = pd.read_html('https://www.geonames.org/postal-codes/DE/BY/bayern.html')
df=dfs[2] # index 2 to import only first table from the list
df.head()


# In[972]:


#group even and odd rows
df = df.groupby(df.index // 2).agg(lambda x: x.dropna().astype(str).str.cat(sep=' '))

#drop unnecessary columns
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

#keep only City = Munich
df = df[df.City == 'Munich']
df= df[['City','Postal Code','Latitude','Longitude']]
df.head()


# ### DATASET 2 (df2) - Average price Monthly Rents for every Postal Code/Districts

# In[973]:


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


# In[974]:


#Download CSV file from github repository: Postal_Code_Munich(3).csv
url_temp_2='https://raw.githubusercontent.com/matborg83/Coursera_Capstone_Munich/main/Postal_Code_Munich(3).csv'
temp_2= pd.read_csv(url_temp_2)

# Normalizing district values to match dataset temp_1
temp_2.loc[85:100,'district'] = 'Schwabing,Schwabing-West'
temp_2 = temp_2.replace(to_replace ='Untergiesing-Harlaching', value = 'Harlaching', regex = True)
temp_2.loc[103:114,'district'] = 'Sendling,Sendling Westpark'
temp_2 = temp_2.replace(to_replace ='Pasing-Obermenzing', value = 'Pasing', regex = True)

temp_2.head()


# In[975]:


df2 = pd.merge(left=temp_1, right=temp_2, left_on='District', right_on='district')
df2= df2[['Postal Code','District','Monthly rent (EUR/Sq meter)']]
df2.head()


# ### DATASET 3 (df3) - All above combined

# In[976]:


df["Postal Code"] = df["Postal Code"].astype(str).astype(int)
df3 = pd.merge(left=df, right=df2, left_on='Postal Code', right_on='Postal Code')
df3= df3[['Postal Code','District','Monthly rent (EUR/Sq meter)','Latitude','Longitude']]

df3.head()


# In[977]:


# Concat Postal Code to create combined Districts with the associated average monthly rent price
temp_3 = df3.sort_values(['Postal Code','District']).groupby('Postal Code', sort=False).District.apply('//'.join).reset_index(name='Combined Districts') 
temp_4= df3.groupby(['Postal Code'],as_index=False).agg({'Monthly rent (EUR/Sq meter)': 'mean'})
temp_5 = pd.merge(left=temp_3, right=temp_4, left_on='Postal Code', right_on='Postal Code')

df3 = pd.merge(left=temp_5, right=df, left_on='Postal Code', right_on='Postal Code')
df3 = df3.drop(['City'], axis=1)
df3.head()


# ### Dataset 4 (Postal_Code_venues_sorted) - 10 main venues for each postal code

# In[978]:


# Define Foursquare Credentials and Version
CLIENT_ID = '3WTP1JKGYST5OV1CAUS3KVOB2I2PGLPTC054P2IF5NWZJYRO' # your Foursquare ID
CLIENT_SECRET = 'HUWVSOQ1NWJEU4TCJ3AMODP0X2KMXHL2UKHSIGUBFR315N2V' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[979]:


#get venues that are in each Postal Code within a radius of 500 meters.

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


# In[980]:


print(Postal_code_venues.shape)
Postal_code_venues.head()


# In[981]:


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


# In[982]:


#Groups the above rows by Postal Code and by taking the mean of the frequency of occurrence of each category
Postal_Code_grouped = munich_onehot.groupby('Postal Code').mean().reset_index()
Postal_Code_grouped.head()


# In[983]:


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
There are many models for clustering. In this project, we will be using the model that is considered one of the simplest models. Despite its simplicity, the K-means is vastly used for clustering in many data science applications, especially useful to quickly discover insights from unlabeled data.
# In[984]:


# set number of clusters
kclusters = 5

#Drop postal code column before running the model.
munich_grouped_clustering = Postal_Code_grouped.drop('Postal Code', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(munich_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[985]:


# add clustering labels
Postal_Code_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

df4 = df3

# merge df4 with Postal_Code_venues_sorted to add latitude/longitude for each neighborhood
df4 = df4.join(Postal_Code_venues_sorted.set_index('Postal Code'), on='Postal Code')

df4.head()


# ### PART 3/ DATA ANALYSING and VIZUALITATION

# In[986]:


# find the mean Monthy rent in EUR for each cluster
df_mean_price = pd.merge(left=df2, right=df4, left_on='Postal Code', right_on='Postal Code')
df_mean_price=df_mean_price[['Postal Code','Monthly rent (EUR/Sq meter)_x','Cluster Labels']]

cluster0 = df_mean_price[df_mean_price['Cluster Labels']==0]
mean_cluster0 = cluster0["Monthly rent (EUR/Sq meter)_x"].mean()

cluster1 = df_mean_price[df_mean_price['Cluster Labels']==1]
mean_cluster1 = cluster1["Monthly rent (EUR/Sq meter)_x"].mean()

cluster2 = df_mean_price[df_mean_price['Cluster Labels']==2]
mean_cluster2 = cluster2["Monthly rent (EUR/Sq meter)_x"].mean()

cluster3 = df_mean_price[df_mean_price['Cluster Labels']==3]
mean_cluster3 = cluster3["Monthly rent (EUR/Sq meter)_x"].mean()

cluster4 = df_mean_price[df_mean_price['Cluster Labels']==4]
mean_cluster4 = cluster5["Monthly rent (EUR/Sq meter)_x"].mean()

print('cluster 0 : ', mean_cluster1)
print('cluster 1 : ', mean_cluster2)
print('cluster 2 : ', mean_cluster3)
print('cluster 3 : ', mean_cluster4)
print('cluster 4 : ', mean_cluster5)


# In[987]:


# On average how many venues were returned per Postal Code in each clusters  HERE TO MODIFY
df_mean_number_venues = pd.merge(left=Postal_code_venues, right=df4, left_on='Postal Code', right_on='Postal Code')
df_mean_number_venues=df_mean_number_venues[['Cluster Labels','Venue']]
df_mean_number_venues.groupby('Cluster Labels').count()


# ### CLUSTER 0 : // Avg price: 21.59EUR // 18 venues // Transportations, restaurants, fast foods and farmer markets

# In[988]:


df4.loc[df4['Cluster Labels'] == 0, df4.columns[[0] + list(range(5, df4.shape[1]))]]


# ### CLUSTER 1: // Avg price: 21.33EUR // 129 venues // Transportations, restaurants, supermarket, bakery, banks

# In[989]:


df4.loc[df4['Cluster Labels'] == 1, df4.columns[[0] + list(range(5, df4.shape[1]))]]


# ### CLUSTER 2: // Avg price: 23.43EUR // 624 venues // Cafes, bars and restaurants

# In[990]:


df4.loc[df4['Cluster Labels'] == 2, df4.columns[[0] + list(range(5, df4.shape[1]))]]


# ### CLUSTER 3: Outlier

# In[991]:


df4.loc[df4['Cluster Labels'] == 3, df4.columns[[0] + list(range(5, df4.shape[1]))]]


# ### CLUSTER 4: // Avg price: 22.32EUR // 898 venues // Multiple Social Venues, Accomodations

# In[992]:


df4.loc[df4['Cluster Labels'] == 4, df4.columns[[0] + list(range(5, df4.shape[1]))]]


# In[993]:


#Obtain Munich Latitude and Longitude with API Geolocator
address = 'Munich, DE'

geolocator = Nominatim(user_agent="TR_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Munich are {}, {}.'.format(latitude, longitude))

#modify df type for Lat and Long
df4['Latitude']= df4['Latitude'].astype(object).astype(float)
df4['Longitude']= df4['Longitude'].astype(object).astype(float)

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
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster) , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
    
map_clusters

