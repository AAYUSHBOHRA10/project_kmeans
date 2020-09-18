#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas_datareader.data as web
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime


# In[12]:


# define instruments to download
companies_dict = {
 'Amazon': 'AMZN',
 'Apple': 'AAPL',
 'Walgreen': 'WBA',
 'Northrop Grumman': 'NOC',
 'Boeing': 'BA',
 'Lockheed Martin':'LMT',
 'McDonalds': 'MCD',
 'Intel': 'INTC',
 'Navistar': 'NAV',
 'IBM': 'IBM',
 'Texas Instruments': 'TXN',
 'MasterCard': 'MA',
 'Microsoft': 'MSFT',
 'General Electric': 'GE',
 'Symantec': 'SYMC',
 'American Express': 'AXP',
 'Pepsi': 'PEP',
 'Coca Cola': 'KO',
 'Johnson & Johnson': 'JNJ',
 'Toyota': 'TM',
 'Honda': 'HMC',
 'Mitsubishi': 'MSBHY',
 'Sony': 'SNE',
 'Exxon': 'XOM',
 'Chevron': 'CVX',
 'Valero Energy': 'VLO',
 'Ford': 'F',
 'Bank of America': 'BAC'
}
companies = sorted(companies_dict.items(), key=lambda x: x[1])


# In[13]:


# Define which online source to use
data_source = 'yahoo'

#define start and end dates
start_date = '2017-01-01'
end_date = '2019-01-01'

# Use pandas_datareader.data.DataReader to load the desired data list(companies_dict.values()) used for python 3 compatibility
panel_data = web.DataReader(list(companies_dict.values()), data_source, start_date, end_date)

print(panel_data.axes)


# In[14]:


# Find Stock Open and Close Values
stock_close = panel_data['Close']
stock_open = panel_data['Open']

print(stock_close.iloc[0])


# In[15]:


# Calculate daily stock movement
stock_close = np.array(stock_close).T
stock_open = np.array(stock_open).T

row, col = stock_close.shape
# create movements dataset filled with 0's

movements = np.zeros([row, col])

for i in range(0, row):
    movements[i, :] = np.subtract(stock_close[i, :], stock_open[i, :])


# In[16]:


for i in range(0, len(companies)):
    print('Company: {}, change: {}'.format(companies[i][0], sum(movements[i][:])))


# In[20]:


plt.figure(figsize=(18, 16))
ax1 = plt.subplot(221)
plt.plot(movements[0][:])
plt.title(companies[0])

plt.subplot(222, sharey=ax1)
plt.plot(movements[1][:])
plt.title(companies[1])
plt.show()


# In[19]:


#import Normalizer
from sklearn.preprocessing import Normalizer
#create the Normalizer
normalizer = Normalizer()

new = normalizer.fit_transform(movements)
print(new.max())
print(new.min())
print(new.mean())


# In[23]:


plt.figure(figsize=(18, 16))
ax1 = plt.subplot(221)
plt.plot(new[0][:])
plt.title(companies[0])

plt.subplot(222, sharey=ax1)
plt.plot(new[1][:])
plt.title(companies[1])
plt.show()


# In[24]:


#import machine learning libraries
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

#define normalizer
normalizer = Normalizer()

#create a K-means model with 10 clusters
kmeans = KMeans(n_clusters=10, max_iter=1000)

# make a pipeline chaining normalizer and kmeans 
pipeline = make_pipeline(normalizer, kmeans)


# In[25]:


# fit pipeline to daily stock movements 
pipeline.fit(movements)


# In[26]:


print(kmeans.inertia_)


# In[27]:


#predict cluster labels
labels = pipeline.predict(movements)

#create a DataFrame aligning labels & companies
df = pd.DataFrame({'labels' : labels, 'companies' : companies})

#display df sorted by cluster labels 
print(df.sort_values('labels'))


# In[28]:


from sklearn.decomposition import PCA
reduced_data = PCA(n_components = 2).fit_transform(new)
kmeans = KMeans(n_clusters = 10)
kmeans.fit(reduced_data)
labels = kmeans.predict(reduced_data)
df = pd.DataFrame({'labels' : labels, 'companies' : companies})
print(df.sort_values('labels'))


# In[37]:


h = 0.01
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap = plt.cm.Paired
plt.clf()
plt.figure(figsize=(10, 10))
plt.imshow(Z, interpolation='nearest', extent = (xx.min(), xx.max(), yy.min(), yy.max()), cmap = cmap, aspect = 'auto', origin = 'lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 169, linewidth = 3, color = 'w', zorder = 10)
plt.title('K-Means Clustering on Stock Market Movements (PCA-Reduced Data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()


# In[ ]:




