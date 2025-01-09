#!/usr/bin/env python
# coding: utf-8

# # Title

# ## Main Objective

# ### Identify Key Drivers of Traffic Safety
# * Use dimensionality reduction PCA to identify the features that have the most variance and influence safety indices.
# ### Monitor Regional Safety Evolution Over Time
# * Group data by region ("Entidad") and year ("Año").
# * Apply clustering (K-Means and others) to group regions based on safety indices.  
#   Combine clustering results with temporal data to observe how cluster assignments change over time.

# ## Dataset Description

# This dataset was obtained from the open data page of Gobierno de México since it was sourced as a PDF's it was extracted in the DataExtraction Notebook that can be accessed trough the following link:  
# *   https://github.com/Arniquin/ClusteringTrafficData  
# 
# ##### Dataset Source:  
# The dataset contains accidents statistics and geographic information per state in Mexico from the year 2018 to 2022.  
# * https://datos.gob.mx/busca/dataset/estadistica-de-accidentes-de-transito

# ### Dataset Features Description
# 
# 1. **Entidad**: Administrative region or state.  
#    *Type*: Categorical. Example: "Mexico City."
# 
# 2. **Superficie**: Region's area in square kilometers (km²).  
#    *Type*: Numerical. Example: 1256.7.
# 
# 3. **Habitantes**: Total population.  
#    *Type*: Numerical. Example: 123456.
# 
# 4. **Hombres**: Male population.  
#    *Type*: Numerical. Example: 60000.
# 
# 5. **Mujeres**: Female population.  
#    *Type*: Numerical. Example: 63500.
# 
# 6. **Densidad de población**: Population density (persons/km²).  
#    *Type*: Numerical. Example: 120.5.
# 
# 7. **Vehículos registrados**: Registered vehicles.  
#    *Type*: Numerical. Example: 45678.
# 
# 8. **Habitantes por vehículo**: Average inhabitants per vehicle.  
#    *Type*: Numerical. Example: 3.5.
# 
# 9. **Índice de motorización**: Vehicles per 1000 inhabitants.  
#    *Type*: Numerical. Example: 280.7.
# 
# 10. **Longitud del camino (km)**: Road length in kilometers (km).  
#     *Type*: Numerical. Example: 150.0.
# 
# 11. **Veh-km (millones)**: Vehicle kilometers traveled (millions).  
#     *Type*: Numerical. Example: 45.6.
# 
# 12. **Accidentes Totales**: Total traffic accidents reported.  
#     *Type*: Numerical. Example: 2000.
# 
# 13. **Accidentes Con muertos**: Traffic accidents with fatalities.  
#     *Type*: Numerical. Example: 45.
# 
# 14. **Accidentes Solo con heridos**: Traffic accidents causing injuries only.  
#     *Type*: Numerical. Example: 300.
# 
# 15. **Accidentes Equivalentes**: Weighted accident count considering severity.  
#     *Type*: Numerical. Example: 500.5.
# 
# 16. **Saldos Muertos**: Total fatalities from traffic accidents.  
#     *Type*: Numerical. Example: 100.
# 
# 17. **Saldos Heridos**: Total injuries from traffic accidents.  
#     *Type*: Numerical. Example: 500.
# 
# 18. **Daños materiales (millones)**: Material damages caused by accidents (millions).  
#     *Type*: Numerical. Example: 12.5.
# 
# 19. **Índices Accidentes por 10⁶ de Veh-km**: Accidents per million vehicle kilometers.  
#     *Type*: Numerical. Example: 3.5.
# 
# 20. **Índices Peligrosidad por 10⁵ de Veh-km**: Danger index per 100,000 vehicle kilometers.  
#     *Type*: Numerical. Example: 2.1.
# 
# 21. **Índices Accidentes mortales por 10⁵ de Veh-km**: Fatal accidents per 100,000 vehicle kilometers.  
#     *Type*: Numerical. Example: 0.15.
# 
# 22. **Índices Muertos por 10⁵ de Veh-km**: Fatalities per 100,000 vehicle kilometers.  
#     *Type*: Numerical. Example: 0.25.
# 
# 23. **Índices Heridos por 10⁵ de Veh-km**: Injuries per 100,000 vehicle kilometers.  
#     *Type*: Numerical. Example: 1.8.
# 
# 24. **Año**: Year of data observation.  
#     *Type*: Temporal. Example: 2020.
# 

# #### Necesary libraries installation and importation

# In[79]:


# Installing the necessary libraries
get_ipython().run_line_magic('pip', 'install pandas scikit-learn seaborn matplotlib')


# In[80]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ### Loading the dataset using pandas read_csv function

# In[81]:


df = pd.read_csv("./data/accidentes.csv")
df.head()


# In[82]:


# Creating a copy of the dataset for modification and analysis
df_copy = df.copy()


# ## Data Cleaning

# There is no data cleaning needed for this dataset since most of the data inconsitencies where handled during the extraction phase and there are no missing values on this dataset.  
# The only thing i will do is to convert the data to numeric values since they are Strings because the data was extracted from PDF's.  
# To do this i will have to replace all of the comas with '' to be able to do the conversion from string to numeric.

# In[83]:


# Replaceing the comas with dots before numeric conversion
df_copy = df_copy.replace({',': ''}, regex=True)


# In[84]:


# Numeric conversion excluding the column Entidad
exclude_column = 'Entidad'
for col in df_copy.columns:
    if col != exclude_column:
        df_copy[col] = pd.to_numeric(df_copy[col])


# ## Exploratory Data Analysis

# In[85]:


df_copy.info()


# In[86]:


df_copy.describe()


# In[87]:


df_copy.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Features", y=1.02)
plt.subplots_adjust(hspace=0.4, wspace=0.6)  


# In[88]:


numeric_columns = df_copy.select_dtypes(include=['float64', 'int64']).drop(columns = ['Año'])
skew_values = numeric_columns.skew()
print(skew_values)


# ## Feature engineering

# Since some absolute values can be misleading when comparing regions with different pipulation sizes i wil be creating some ratios to standardize the data this way allowing fare comparison between regions.  
# This also helps for identifying trends between regions.  
# For this reasons i will create a Accidents per capita and a Vehicles per capita features allowing to identify the regions  where residents are at a higher risk of being involved in accidents and the level of motorization.

# In[89]:


df_copy['Accidents Per Capita'] = df_copy['Accidentes Totales'] / df_copy['Habitantes']
df_copy['Vehicles Per Capita'] = df_copy['Vehículos registrados'] / df_copy['Habitantes']
numeric_columns = df_copy.select_dtypes(include=['float64', 'int64']).drop(columns = ['Año'])
df_copy[['Accidents Per Capita', 'Vehicles Per Capita']]



# In[90]:


skew_values = numeric_columns.skew()
high_skew = skew_values[skew_values > 1.5].index
for feature in high_skew:
    df_copy[feature] = np.log1p(df_copy[feature])  


# In[91]:


df_copy.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Features", y=1.02)
plt.subplots_adjust(hspace=0.4, wspace=0.6)  


# In[92]:


numeric_columns = df_copy.select_dtypes(include=['float64', 'int64']).drop(columns = ['Año'])
skew_values = numeric_columns.skew()
print(skew_values)


# Next i will normalize my features to ensure equal weighting in dimensionality reduction and clustering utilizing sklearn's StandardScaler

# In[93]:


scaler = StandardScaler()
numeric_features = df_copy.select_dtypes(include=['float64', 'int64']).columns
df_copy[numeric_features] = scaler.fit_transform(df_copy[numeric_features])


# In[94]:


model_df = df_copy.drop(['Entidad', 'Año'], axis=1)  # Drop irrelevant columns


# In[95]:


model_df.head()


# Finally for feature engineering i will create a summary statistics of some key features to apply clustering over time.

# In[96]:


region_summary = df_copy.groupby('Entidad').agg({
    'Accidentes Totales': 'sum',
    'Habitantes': 'mean',
    'Vehículos registrados': 'mean',
    'Índices Accidentes por 10^6 de Veh-km': 'mean'
}).reset_index()


# ## Modeling

# ### PCA

# In[97]:


from sklearn.decomposition import PCA

pca = PCA(n_components=0.9)  # Retain 90% of variance
pca_components = pca.fit_transform(model_df)


# In[98]:


pca.explained_variance_ratio_


# In[99]:


pca.components_


# ### Clustering

# In[100]:


# Kmeans
from sklearn.cluster import KMeans

wcss = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(model_df)
    wcss.append(kmeans.inertia_)  # WCSS for this k



# In[101]:


plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.show()


# In[106]:


kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(model_df)
df['Cluster'] = clusters 


# In[115]:


# DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.001, min_samples=4)
dbscan_clusters = dbscan.fit_predict(model_df)
df_copy['DBSCAN_Cluster'] = dbscan_clusters


# ### Results Analysis

# In[116]:


import matplotlib.pyplot as plt

plt.scatter(pca_components[:, 0], pca_components[:, 1], c=dbscan_clusters, cmap='viridis')
plt.colorbar()
plt.title('PCA with Clustering')
plt.show()


# ## Model Recomendation

# ## Key Findings and Insights

# ## Next Steps
