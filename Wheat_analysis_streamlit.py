import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the data
cols = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove', 'class']
dp = pd.read_csv('seeds_dataset.txt', names=cols, sep='\s+')

# Display the dataset
st.write("### Dataset")
st.write(dp.head())

# Visualize scatterplots
st.write("### Scatterplots")
for i in range(len(cols)-1):
    for j in range(i+1, len(cols)-1):
        x_label = cols[i]
        y_label = cols[j]
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_label, y=y_label, data=dp, hue='class', ax=ax)
        st.pyplot(fig)

# Clustering
st.write("### Clustering")
x = 'compactness'
y = 'asymmetry'
x1 = dp[[x, y]].values
kmeans = KMeans(n_clusters=3).fit(x1)
clusters = kmeans.labels_
cluster_dp = pd.DataFrame(np.hstack((x1, clusters.reshape(-1,1))), columns=[x, y, 'class'])

# Visualize clustering results
st.write("#### KMeans Clustering")
fig, ax = plt.subplots()
sns.scatterplot(x=x, y=y, data=cluster_dp, hue='class', ax=ax)
st.pyplot(fig)

# PCA
st.write("### PCA")
x2 = dp[cols[:-1]].values
pca = PCA(n_components=2)
transformed_x = pca.fit_transform(x2)
kmeans_pca_dp = pd.DataFrame(np.hstack((transformed_x, clusters.reshape(-1,1))), columns=['pca1', 'pca2', 'class'])

# Visualize PCA results
st.write("#### KMeans PCA")
fig, ax = plt.subplots()
sns.scatterplot(x='pca1', y='pca2', data=kmeans_pca_dp, hue='class', ax=ax)
st.pyplot(fig)
