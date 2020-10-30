#!/usr/bin/env python
# coding: utf-8

# # **Housing Sales Prices & Venues Data Analysis of Munich
# 

# ## PART 1/DATA SCRAPING AND CLEANING

# ### DATASET 1

# In[ ]:


# Determining where is the best to leave in Munich
Segmenting and Clustering Venues in Neighborhoods of Munich's
assumption: the person like this specific kind of place and he is looking for a mid HPI
Dataset:


# In[199]:


#Install required Library
import pandas as pd
import numpy as np
import re #Python RegEx


# In[221]:


#Get the data from geonames website into Pandas DF
dfs = pd.read_html('https://www.geonames.org/postal-codes/DE/BY/bayern.html')
df=dfs[2] # index 2 to import only first table from the list
df


# In[222]:


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
df


# ### DATASET 2

# Dowload CSV House pricing dataset  do try for free or other IP to view and put inside csv plus write the coresponding postcode
# extract the CSV File and add postcode
# then merge the two table and define lo, mid, high prices
# 
# collected and modified in Excel
# dataset1
# https://www.statista.com/statistics/800552/rental-costs-in-munich-germany-by-district/
# dataset2
# https://www.muenchen.de/int/en/living/postal-codes.html
# 

# In[241]:


url2='https://raw.githubusercontent.com/matborg83/Coursera_Capstone_Munich/main/Average_apart_rent_Munich.csv'
df2 = pd.read_csv(url)
df2


# In[263]:


url3='https://raw.githubusercontent.com/matborg83/Coursera_Capstone_Munich/main/Postal_Code_Munich(3).csv'
df3= pd.read_csv(url3)
df3


# In[ ]:


python do operation replace for follozing values on df3 table then match two tables
Schwabing,Schwabing-West
pasing
Untergiesing-Harlaching
munchen - to all the one left
Neuhausen, Nymphenburg
sendling

