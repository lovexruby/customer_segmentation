import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("../data/mall_customers.csv")
# Test overview
print(df.head())
print(df.info())

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize features (finally --> mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
# Use the Elbow Method to determine the optional number of clusters
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
"""



# Initialize K-Means with calculated 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Assign cluster labels to original dataframe
df['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(8,6))
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis', s=50)
#plt.plot(k_range, inertia, marker='o')
#sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', data=df)
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
#plt.tight_layout
plt.savefig('../plots/customer_segments.png')
plt.show()