# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:38:25 2021

@author: Shahrukh
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn

pd.set_option('display.max_column',None)
retail_df=pd.read_csv('Online+Retail.csv')
retail_df.info()
retail_df.head()

#Missing Value in each column 

retail_df.isnull().sum()
#Missing value in term of perccetnages
100*((retail_df.isnull().sum())/len(retail_df))


#drop all rows having missing value
retail_df=retail_df.dropna()
100*((retail_df.isnull().sum())/len(retail_df))


#Prepare the data for modelling
#RFM Analysis
#R-Recency- Number of days since last purcahse
#F-Number of Transaction/year
#M-(Monetary) :Total amount of purchase


#Monetaray

#New Column Amount
retail_df['Amount']=retail_df['Quantity']*retail_df['UnitPrice']

#Amount spend by each customer
s=retail_df.groupby('CustomerID')['Amount'].sum()
#REturns Customer ID as index 
#Convert it into column
s=s.reset_index()

#Frequency
frequency_df=retail_df.groupby('CustomerID')['InvoiceNo'].count()
frequency_df=frequency_df.reset_index()

#Change Column NAmes

frequency_df.columns=['CustomerID','Frequency']

#Merge Two DF
grouped_df=pd.merge(s,frequency_df,on='CustomerID',how='inner')

#recency numebr of days since last transaction

retail_df['InvoiceDate']=pd.to_datetime(retail_df['InvoiceDate'],
                                     format='%d-%m-%Y %H:%M'
                                        )

#compute the max date
max_date=max(retail_df['InvoiceDate'])
#compute the diff
retail_df['diff']=max_date-retail_df['InvoiceDate']

#recency column

last_purchase=retail_df.groupby('CustomerID')['diff'].min()

last_purchase=last_purchase.reset_index()

grouped_df=pd.merge(grouped_df,last_purchase,on='CustomerID',how='inner')

grouped_df.columns=['CustomerID','Amount','Frequency','Recency']

grouped_df['Recency']=grouped_df['Recency'].dt.days


#Outlier Treatment
#Rescaling Variables

rfm_df=grouped_df[['Amount','Frequency','Recency']]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
rfm_df_scaled=scaler.fit_transform(rfm_df)
rfm_df_scaled

rfm_df_scaled=pd.DataFrame(rfm_df_scaled)

rfm_df_scaled.columns=['Amount','Frequency','Recency']
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=4,max_iter=50)

range_n_clusters=[2,3,4,5,6,7,8]
ssd=[]
#SSD Metrics
for i in range_n_clusters:
    kmeans=KMeans(n_clusters=i,max_iter=50)
    kmeans.fit(rfm_df_scaled)
    ssd.append(kmeans.inertia_)
plt.plot(ssd)


#Sillhoute Analysis
from sklearn.metrics import silhouette_score
range_n_clusters=[2,3,4,5,6,7,8]
for i in range_n_clusters:
    kmeans=KMeans(n_clusters=i,max_iter=50)
    kmeans.fit(rfm_df_scaled)
    #sillhouteSCore
    cluster_labels=kmeans.labels_
    s_avg=silhouette_score(rfm_df_scaled,cluster_labels)
    print('the silhoute score for cluster{0} is {1}'.format(i,s_avg))

#Build the model with cluster =3


kmeans=KMeans(3,max_iter=50)
kmeans.fit(rfm_df_scaled)


#labels=kmeans.labels_
grouped_df['clusterID']=kmeans.labels_

#number of customers in 1,2,& 3 clsuters
grouped_df['clusterID'].value_counts()













