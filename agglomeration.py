import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from fastapi import FastAPI, HTTPException, Request
import os
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle
import uvicorn

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ---------------------------------- Data Processing --------------------------------
# The data process set X on : 'Income', 'Score' and Y on : 'Gender'
                              
df = pd.read_csv(r'./data/Mall_Customers.csv')
df.rename(index=str, columns={'Annual Income (k$)': 'Income',
                              'Spending Score (1-100)': 'Score'}, inplace=True)

X = df.drop(['CustomerID', 'Gender'], axis=1)
print("Data Processing validated")




# ---------------------------------- Model and Params --------------------------------
n_clusters = 5
agglom = AgglomerativeClustering(n_clusters, linkage='average').fit(X)
 

X['Labels'] = agglom.labels_

print("Model and Params validated")



# ---------------------------------- Plot --------------------------------
def plot_agglo(cluster_label, n_clusters, cluster_points):
    # Plot the clusters
    plt.figure(figsize=(12, 8))
    for cluster_label in range(n_clusters):  
        # Extract points belonging to the current cluster
        cluster_points = X[X['Labels'] == cluster_label]
        
        # Calculate the centroid as the mean position of the data points in the cluster
        centroid = cluster_points[['Income', 'Score']].mean(axis=0)
        
        # Plot points for the current cluster
        plt.scatter(cluster_points['Income'], cluster_points['Score'],
                    s=50, label=f'Cluster {cluster_label + 1}')
        
        # Plot the centroid
        plt.scatter(centroid[0], centroid[1], s=300, c='black', marker='*', label=f'Centroid {cluster_label + 1}')

    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.savefig('plot_agglo.png')

    print("Plot validated")
# ---------------------------------- Metrics --------------------------------

# Calculate silhouette score

#def calc_silhouette(X, agglom.labels_):
#    silhouette = silhouette_score(X[['Income', 'Score']], agglom.labels_)
#    print(f"Silhouette Score: {silhouette}")
#    return silhouette

# Calculate Davies–Bouldin index
#def calc_davies_bouldin(X, agglom.labels_):
#    davies_bouldin = davies_bouldin_score(X[['Income', 'Score']], agglom.labels_)
#    print(f"Davies–Bouldin Index: {davies_bouldin}")
#    return davies_bouldin




# ---------------------------------- API --------------------------------
app = FastAPI()
model = AgglomerativeClustering
model_name = f"{model}"

@app.post('/prediction_agglo')
async def prediction_agglo(n_clusters : int):
    agglom = model(n_clusters, linkage='average').fit(X)
    # davies_bouldin metric
    silhouette = silhouette_score(X[['Income', 'Score']], agglom.labels_)
    print(f"Silhouette Score: {silhouette}")
    # davies_bouldin metric
    davies_bouldin = davies_bouldin_score(X[['Income', 'Score']], agglom.labels_)
    print(f"Davies–Bouldin Index: {davies_bouldin}")
    print(f"Prediction : {agglom} effectued with {model_name}")
    return f"Prediction : {agglom} effectued with {model_name}"
    
print("Plot and Metrics sended")


# ---------------------------------- Model Export --------------------------------

pickle.dump(agglom, open('model_agglo.pkl', 'wb'))

print("Model Export validated")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)