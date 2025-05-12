import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import zipfile
import os

#Stocks from the Dow Jones
zip_path = "/Users/devinreed/Downloads/INST414/Module4/Dow_Data.zip"
folder_data = "./dow_data"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(folder_data)

Dow_data = pd.DataFrame()


for file in os.listdir(folder_data):
    if file.endswith(".csv"):
        path = os.path.join(folder_data, file)
        df = pd.read_csv(path)


        if 'date' in df.columns and 'close' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            ticker = file.replace(".csv", "")
            Dow_data[ticker] = df['close']



#daily returns of stocks on the DOW
Dow_data.dropna(axis=0, inplace=True)
dailyReturns = Dow_data.pct_change().dropna()


metrics = pd.DataFrame({
    'Average Return': dailyReturns.mean(),
    'Volatility': dailyReturns.std()
}).dropna()



scaler = StandardScaler()
metrics_scaled = scaler.fit_transform(metrics)

#searches for best k value 1-10
sum_of_sq = []
k_range = range(1, 10)
for k in k_range:
    kmean = KMeans(n_clusters = k, random_state = 42)
    kmean.fit(metrics_scaled)
    sum_of_sq.append(kmean.inertia_)

#plot elbow curve
plt.figure(figsize=(8, 4))
plt.plot(k_range, sum_of_sq, 'o-')
plt.xlabel('Clusters')
plt.ylabel('Sum of Squares')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

#optimal k based on elbow
k = 4
kmeans = KMeans(n_clusters=k, random_state=40)
metrics['Cluster'] = kmeans.fit_predict(metrics_scaled)


#plot clusters
plt.figure(figsize=(10, 6))
for cluster in range(k):
    group = metrics[metrics['Cluster'] == cluster]
    plt.scatter(group['Volatility'], group['Average Return'], label=f'Cluster {cluster}')
    for ticker in group.index:
        plt.annotate(ticker, (group.loc[ticker, 'Volatility'], group.loc[ticker, 'Average Return']), fontsize=8)

plt.title('Dow 30 Stock Clusters')
plt.xlabel('Volatility')
plt.ylabel('Average Daily Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()









