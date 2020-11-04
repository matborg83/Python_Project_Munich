#!/usr/bin/env python
# coding: utf-8

# # **Rent Prices & Nearby Venues Data Analysis of Munich
# 

# In[ ]:


Determine where is the best to leave in Munich (depending on the cost and nearby venues)
Segmenting and Clustering Venues in Neighborhoods of Munich's
assumption: the person like this specific kind of place and he is looking for a mid HPI
Dataset:yy

Part 1:
Part 2:
Part 3:


# ## PART 1/ DATA SCRAPING and CLEANING

# ### DATASET 1 - Postal Code, Latitude and Longitude

# In[506]:


#Install required Library
import pandas as pd
import numpy as np
import re #Python RegEx


# In[507]:


#Get the data from geonames website into Pandas DF
dfs = pd.read_html('https://www.geonames.org/postal-codes/DE/BY/bayern.html')
df=dfs[2] # index 2 to import only first table from the list
df.head()


# In[508]:


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

# In[509]:


# Dataset 2 contains combined dataset temp_1 and temp_2 dowloaded from the project github repository.

#dataset1
#original source: https://www.statista.com/statistics/800552/rental-costs-in-munich-germany-by-district/

#dataset2
#original source: https://www.muenchen.de/int/en/living/postal-codes.html


# In[510]:


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


# In[511]:


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


# In[512]:


df2 = pd.merge(left=temp_1, right=temp_2, left_on='District', right_on='district')
df2= df2[['Postal Code','District','Monthly rent (EUR/Sq meter)']]
df2.head()


# ### DATASET 3 - All above combined

# In[513]:


df["Postal Code"] = df["Postal Code"].astype(str).astype(int)
df3 = pd.merge(left=df, right=df2, left_on='Postal Code', right_on='Postal Code')
df3= df3[['Postal Code','District','Monthly rent (EUR/Sq meter)','Latitude','Longitude']]

df3.head()


# In[514]:


# Concat Postal Code to create new unique Postal Code with combined Districts and average monthly rent price 
temp_3 = df3.sort_values(['Postal Code','District']).groupby('Postal Code', sort=False).District.apply('//'.join).reset_index(name='Combined Districts') 
temp_4= df3.groupby(['Postal Code'],as_index=False).agg({'Monthly rent (EUR/Sq meter)': 'mean'})
temp_5 = pd.merge(left=temp_3, right=temp_4, left_on='Postal Code', right_on='Postal Code')

df3 = pd.merge(left=temp_5, right=df, left_on='Postal Code', right_on='Postal Code')
df3 = df3.drop(['City'], axis=1)
df3.head()


# ### Dataset 4 - 10 main venues by postal code using Foursquare API

# In[515]:


#import additional library
import requests


# In[516]:


# Define Foursquare Credentials and Version
CLIENT_ID = '3WTP1JKGYST5OV1CAUS3KVOB2I2PGLPTC054P2IF5NWZJYRO' # your Foursquare ID
CLIENT_SECRET = 'HUWVSOQ1NWJEU4TCJ3AMODP0X2KMXHL2UKHSIGUBFR315N2V' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[517]:


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


# In[518]:


print(Postal_code_venues.shape)
Postal_code_venues.head()


# In[519]:


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


# In[520]:


#Groups the above rows by Postal Code and by taking the mean of the frequency of occurrence of each category
Postal_Code_grouped = munich_onehot.groupby('Postal Code').mean().reset_index()
Postal_Code_grouped.head()


# In[521]:


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


# ## PART 2/ CLUSTERING - MACHINE LEARNING -

# In[522]:


#import library for clustering
from sklearn.cluster import KMeans


# In[523]:


#create seaborn to see where is k maximized then modified the k below





# In[524]:


# set number of clusters
kclusters = 5

munich_grouped_clustering = Postal_Code_grouped.drop('Postal Code', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(munich_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[525]:


# add clustering labels
Postal_Code_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
Postal_Code_venues_sorted.head()

df4 = df3

# merge manhattan_grouped with manhattan_data to add latitude/longitude for each neighborhood
df4 = df4.join(Postal_Code_venues_sorted.set_index('Postal Code'), on='Postal Code')

df4.head()


# 

# In[ ]:


manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 0, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]


# In[ ]:


#examine and classify the cluster


# ### Part 3 - DATA ANALYSING and VIZUALITATION

# In[486]:





# In[487]:


# how many unique categories can be curated from all the returned venues
print('There are {} uniques categories.'.format(len(Postal_code_venues['Venue Category'].unique())))
# how many venues were returned for each Postal Code
Postal_code_venues.groupby('Postal Code').count()


#Add 5 most pop avenue per postal code and the cluster number see from -  3. Analyze Each Neighborhood


# In[ ]:


# visualize the resulting clusters on map and add chore map for rent price


# In[ ]:




