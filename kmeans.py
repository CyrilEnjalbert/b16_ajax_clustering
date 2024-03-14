import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import os
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

df = pd.read_csv(r'./data/Mall_Customers.csv')
df.columns = ['customerID', 'gender', 'age','income', 'score']

x=pd.DataFrame(df.score)
y=pd.DataFrame(df.age)

sns.scatterplot(X, x= "income", y= "score")
plt.xlabel('Age'), plt.ylabel('annual income') 
plt.title('age vs income')
plt.legend()
plt.show()

X = df[['income', 'score']]

# Feature normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(x)
n_cluster = 5
random_state = 42

kmeans = KMeans(n_clusters=n_cluster, init='k-means++', random_state=random_state)
kmeans.fit(X_scaled)
plt.figure(figsize=(10, 8))
for cluster_label in range(5):  # Loop through each cluster label
    cluster_points = X[kmeans.labels_ == cluster_label]
    centroid = cluster_points.mean(axis=0)  # Calculate the centroid as the mean position of the data points
    plt.scatter(cluster_points['income'], cluster_points['score'],
                s=50, label=f'Cluster {cluster_label + 1}')  # Plot points for the current cluster
    plt.scatter(centroid[0], centroid[1], s=300, c='black', marker='*', label=f'Centroid {cluster_label + 1}')  # Plot the centroid
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('plot_kmeans.png')
plt.show()

# import pickle
# pickle.dump(kmeans, open('model_kmeans', 'wb'))
