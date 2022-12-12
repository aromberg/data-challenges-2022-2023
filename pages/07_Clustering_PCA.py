import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
st.markdown(
"""
# Clustering PCA Data
## Preprocess Data
Since PCA can only handle numeric data, we extract all numeric columns from the
dataset.
""")
data = pd.read_csv("data/data_imp.csv")
data = data[["age","trestbps", "chol", "thalach", "oldpeak"]]
st.dataframe(data.head())

st.markdown(
"""
## Principal Component Analysis
We do a PCA on the numerical data.
"""
)
pca = PCA(n_components=5)
pca_data = pca.fit_transform(data)
pca_data = pd.DataFrame(pca_data)
x_axis = st.selectbox("Please select the PC displayed on the x-axis.",
pca_data.columns)
y_axis = st.selectbox("Please select the PC displayed on the y-axis.",
pca_data.columns, index=1)
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
fig = plt.figure()
p = sns.scatterplot(data=pca_data, x=pca_data[x_axis], y=pca_data[y_axis],
palette="deep")
st.pyplot(fig)

st.markdown(
"""
## K-Means
We cluster the PCA data using the K-Means algorithm.
"""
)
k = st.slider('Please select the number of clusters.', min_value=1, max_value=8)
kmeans = KMeans(n_clusters=k).fit(pca_data)
labels = kmeans.predict(pca_data)
fig = plt.figure()
p = sns.scatterplot(data=pca_data, x=pca_data[x_axis], y=pca_data[y_axis],
hue=labels, palette="deep")
st.pyplot(fig)

st.markdown(
"""
## DBSCAN
We cluster the PCA data using the DBSCAN algorithm.
"""
)
eps = st.slider('Please select epsilon.', min_value=10, max_value=30)
min_samples = st.slider('Please select the minimal number of samples.',
min_value=5, max_value=10)
clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(pca_data)
num_values = len(set(clusters.labels_))
if num_values < 9 :
    with_legend = True
else :
    with_legend = False
fig = plt.figure()
p = sns.scatterplot(data=pca_data, x=pca_data[x_axis], y=pca_data[y_axis],
hue=clusters.labels_, palette="deep", legend=with_legend)
st.pyplot(fig)

st.markdown(
"""
## OPTICS
We cluster the PCA data using the OPTICS algorithm.
"""
)
min_samp = st.slider('Please select the minimum number of samples.',
min_value=1, max_value=30, help="The number of samples in a neighborhood for \
a point to be considered as a core point.")
clusters = OPTICS(min_samples=min_samp).fit(pca_data)
num_values = len(set(clusters.labels_))
if num_values < 9 :
    with_legend = True
else :
    with_legend = False
fig = plt.figure()
p = sns.scatterplot(data=pca_data, x=pca_data[x_axis], y=pca_data[y_axis],
hue=clusters.labels_, palette="deep", legend=with_legend)
st.pyplot(fig)
