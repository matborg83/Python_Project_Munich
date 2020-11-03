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


# ## PART 1/DATA SCRAPING AND CLEANING

# ### DATASET 1 - Postal Code, Latitude and Longitude

# In[397]:


#Install required Library
import pandas as pd
import numpy as np
import re #Python RegEx


# In[414]:


#Get the data from geonames website into Pandas DF
dfs = pd.read_html('https://www.geonames.org/postal-codes/DE/BY/bayern.html')
df=dfs[2] # index 2 to import only first table from the list
df.head()


# In[415]:


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
df


# ### DATASET 2 - Average District Monthly Rents and Postal Code

# In[421]:


# Dataset 2 contains combined dataset temp_1 and temp_2 dowloaded from the project github repository.

#dataset1
#original source: https://www.statista.com/statistics/800552/rental-costs-in-munich-germany-by-district/

#dataset2
#original source: https://www.muenchen.de/int/en/living/postal-codes.html


# In[416]:


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


# In[418]:


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

temp_2


# In[410]:


df2 = pd.merge(left=temp_1, right=temp_2, left_on='District', right_on='district')
df2= df2[['Postal Code','District','Monthly rent (EUR/Sq meter)']]
df2


# ### DATASET 3 - All data combined

# In[462]:


df["Postal Code"] = df["Postal Code"].astype(str).astype(int)
df3 = pd.merge(left=df, right=df2, left_on='Postal Code', right_on='Postal Code')
df3= df3[['Postal Code','District','Monthly rent (EUR/Sq meter)','Latitude','Longitude']]

df3


# In[461]:


# Concat Postal Code to create new unique Postal Code with combined Districts and average monthly rent price 
temp_3 = df3.sort_values(['Postal Code','District']).groupby('Postal Code', sort=False).District.apply('//'.join).reset_index(name='Combined Districts') 
temp_4= df3.groupby(['Postal Code'],as_index=False).agg({'Monthly rent (EUR/Sq meter)': 'mean'})
temp_5 = pd.merge(left=temp_3, right=temp_4, left_on='Postal Code', right_on='Postal Code')

df3 = pd.merge(left=temp_5, right=df, left_on='Postal Code', right_on='Postal Code')
df3 = df3.drop(['City'], axis=1)
df3


# ## PART 2/DATA VISUALISATION

# ### Part 1 - Clustering and plotting the Postal Code of Munich

# In[ ]:





# In[ ]:





# ### Part 2 - Average District Monthly Rents and Postal Code

# In[ ]:





# In[ ]:





# ### Part 3 - Summary map

# In[ ]:




