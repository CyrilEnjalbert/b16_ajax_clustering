import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import os
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from fastapi import FastAPI, HTTPException, Request

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ---------------------------------- Data Processing --------------------------------
# The data process set X on : 'Income', 'Score' and Y on : 'Gender'.

df = pd.read_csv(r'./data/Mall_Customers.csv')
df.head()

df.dtypes

df.describe()

df.rename(index=str, columns={'Annual Income (k$)': 'Income',
                              'Spending Score (1-100)': 'Score'}, inplace=True)

X = df.drop(['CustomerID', 'Gender'], axis=1)

# Feature normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Data Processing validated")

# ---------------------------------- Model and Params --------------------------------

n_cluster = 5
random_state = 42

kmeans = KMeans(n_clusters=n_cluster, init='k-means++', random_state=random_state)
kmeans.fit(X_scaled)

print("Model and Params validated")

# ---------------------------------- Plot --------------------------------

plt.figure(figsize=(10, 8))
for cluster_label in range(5):  # Loop through each cluster label
    cluster_points = X[kmeans.labels_ == cluster_label]
    centroid = cluster_points.mean(axis=0)  # Calculate the centroid as the mean position of the data points
    plt.scatter(cluster_points['Income'], cluster_points['Score'],
                s=50, label=f'Cluster {cluster_label + 1}')  # Plot points for the current cluster
    plt.scatter(centroid[0], centroid[1], s=300, c='black', marker='*', label=f'Centroid {cluster_label + 1}')  # Plot the centroid
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('plot_kmeans.png')


print("Plot validated")



# ---------------------------------- Metrics --------------------------------

# Calculate silhouette score
silhouette = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score: {silhouette}")

# Calculate Davies–Bouldin index
davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
print(f"Davies–Bouldin Index: {davies_bouldin}")

print("Metrics validated")


# ---------------------------------- API --------------------------------
app = FastAPI()
model = KMeans
model_name = f"{model}"

@app.post('/prediction_kmeans')
async def prediction_kmeans(n_clusters : int):
    kmeans = model(n_clusters, init='k-means++', random_state=random_state)
    kmeans.fit(X_scaled)
    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    print(f"Silhouette Score: {silhouette}")
    # Calculate Davies–Bouldin index
    davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
    print(f"Davies–Bouldin Index: {davies_bouldin}")
    return f"Prediction : {kmeans} effectued with {model_name}"
    
print("Plot and Metrics sended")




# ---------------------------------- Model Export --------------------------------
import pickle

pickle.dump(kmeans, open('model_kmeans.pkl', 'wb'))

print("Model Export validated")

# ---------------------------------- Server --------------------------------
import uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)