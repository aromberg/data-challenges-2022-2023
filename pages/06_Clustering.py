import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import pairwise_distances, davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import gower
import numpy as np


##################### Define Helper functions #################################


def PreProcessing(data, num=False, cat=False, mixed=False):
    """
    Standardize numerical features by removing mean and scaling to unit variance.
    If mixed is True, categorical variables will be turned to one-hot numerical
    array.

    """
    # first divide dataframe into two other dataframes
    cater_columns = list(data.columns[data.dtypes == 'category'])
    cater = data[cater_columns]
    numer_columns = list(data.columns[data.dtypes != 'category'])
    numer = data[numer_columns]
    if num:
    # rescale numerical variables (normalization)
        scaler = StandardScaler()
        numer = pd.DataFrame(scaler.fit_transform(numer))
        numer.columns = [sub + "_scaled" for sub in numer_columns]
        if not cat and not mixed:
            return numer
    if cat:
        transformer = make_column_transformer(
            (OneHotEncoder(), list(cater.columns)),
            remainder='passthrough'
        )
        transformed = transformer.fit_transform(cater)
        cols = transformer.get_feature_names_out()
        cater = pd.DataFrame(transformed, columns=cols)
        if not num and not mixed:
            return cater
    if mixed:
        preproc = pd.concat([numer, cater], axis=1, join="inner")
        return preproc

    preproc_onesided = pd.concat([numer, cater], axis=1, join="inner")
    return preproc_onesided



def SubClusterDF(k, df):
    """
    Creates a list containing sub-dataframes of the dataset.
    Division of the df by membership of cluster.

    Args:
        k: number of clusters created
        df: dataset
    Returns:
        ls: list of sub-dataframes
    """
    ls = []
    for i in range(k):
        cluster = df[df["Clusters"] == i]
        ls.append(cluster)
    return ls


def SubScatterCluster(clusters, cols, xlabel, ylabel):
    """
    Creates scatter objects.

    Args:
        clusters: list of sub-dataframes, each
                    representing a cluster
        cols: list of colors

     Returns:
        ls: list of scatter objects
    """
    ls = []
    j = -1
    for i in range(len(clusters)):
        j += 1
        if j >= len(cols):
            j = 0
        trace = go.Scatter(
            x=clusters[i][xlabel],
            y=clusters[i][ylabel],
            mode="markers",
            name=f"Cluster {i}",
            marker=dict(color=cols[j]),
            text=None
        )
        ls.append(trace)
    return ls


####################### Prepare Data ##########################################

# read file
data = pd.read_csv("data/heart_disease/heart_disease_combined.csv")
# drop coll Unnamed: 0 and rows with NAs
data = data.drop(labels="Unnamed: 0", axis=1)
data = data.dropna(axis=0)

# assign new dtypes
dtyp = {"sex": "category",
        "cp": "category",
        "restecg": "category",
        "fbs": "category",
        "exang": "category",
        "slope": "category",
        "ca": "category",
        "thal": "category",
        "num": "category"}

dat_unmod = data.astype(dtyp)
# create a copy to avoid manipulating the original dataset
dat = dat_unmod.copy()
dat = dat.astype(dtyp)

####################### Define variables ######################################

colors = ["rgba(255, 128, 255, 0.8)",  # pink
          "rgba(255, 128, 2, 0.8)",  # orange
          "rgba(0, 255, 200, 0.8)",  # turquoise
          "rgba(102, 205, 0, 0.8)",  # green
          "rgba(255, 215, 0, 0.8)",  # yellow
          "rgba(123, 104, 238, 0.8)",  # mediumslateblue
          "rgba(51, 161, 201, 0.8)",  # peacock
          "rgba(255, 48, 48, 0.8)",  # firebrick1
          "rgba(65, 105, 225, 0.8)",  #royalblue
          "rgba(238, 99, 99, 0.8)",  # indianred1
          "rgba(180, 82, 205, 0.8)",  # mediumorchid3
          "rgba(84, 255, 159, 0.8)",  #seagreen1
          "rgba(135, 206, 255, 0.8)", # skyblue1
          "rgba(0, 255, 127, 0.8)",  # springgreen
          "rgba(255, 165, 79, 0.8)",  # tan1
          "rgba(255, 0, 0, 0.8)",  # red1
          "rgba(255, 187, 255, 0.8)",  # plum1
          "rgba(255, 218, 185, 0.8)",  # peachpuff1
          "rgba(199, 21, 133, 0.8)",  # mediumvioletred
          "rgba(255, 182, 193, 0.8)", # lightpink
          "rgba(255, 236, 139, 0.8)", # lightgoldenrod1
          "rgba(255, 105, 180, 0.8)"] # hotpink

title = "Visualizing Clusters in Two Dimensions Using PCA"
layout = dict(title=title,
              xaxis= dict(title="PC1", ticklen=5, zeroline=False),
              yaxis=dict(title="PC2", ticklen=5, zeroline=False))


############################### Set Page ##########################################

###############################  K-Means ##########################################

st.set_page_config(page_title="Clustering",
                   page_icon="🗺")

st.markdown(
    """
    # Clustering
    ## K-Means

    The K-Means clustering algorithm is usually applied on numerical data.
    However, our dataset contains both numerical and categorical variables.
    There are K-Means variants such as K-Modes which is sutiable for categorical 
    data or K-Prototype which is able to deal with mixed data.

    For our implementation, we leave it to the user to apply the
    K-Means Algorithm only on numerical variables or both, numerical and categorical
    data to see differences in cluster formation.
    
    If the user chooses to apply K-Means only on numerical data, we provide the
    option to [standardize numerical variabels](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
    Since these variables take incomparable units such as resting blood pressure (trestbps) in mm Hg or
    serum cholestoral (chol) in mg/dl. It is intended to show the differences in
    cluster formation if standardization is applied or not. However, since K-means
    uses the euclidean distance measure, it is [recommended to apply standardization](https://medium.com/geekculture/7-important-distance-metrics-every-data-scientist-should-know-11e1b0b2ebe3).

    If the user wants to apply K-Means on the mixed data, numerical values will
    be standardized on default and categorical variables (nominal and binary) will be encoded as
    [one-hot numeric array](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

    For visualization, we apply PCA after running K-Means to plot clusters in two dimensions. However, it is recomemended
    to [apply dimensionality reduction prior to K-Means](https://mclguide.readthedocs.io/en/latest/sklearn/clusterdim.html).
    """
)

k = st.slider('Number of Clusters', value=3, min_value=1, max_value=8)
init_method = st.selectbox("Initialization Method", ["k-means++", "random"],
                           help="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html")
typ_kmeans = st.selectbox("Variables", ["numerical", "numerical and categorical"],
                   help="If 'numerical and categorical' is choosen, numerical variables will be standardized and categorical features will be encoded as one-hot numeric array")

if typ_kmeans == "numerical":
    stand = st.checkbox("Standardize features", value=True)


if init_method == "k-means++":
    kmeans = KMeans(n_clusters=k, init=init_method)
else:
    kmeans = KMeans(n_clusters=k, init=init_method, random_state=222)
# convert dataframe to dict

if typ_kmeans == "numerical and categorical":
    dat = PreProcessing(data=dat, num=True, cat=True, mixed=True)
else:
    if stand:
        dat = PreProcessing(data=dat, num=True)

# connvert data to dictionary
data_dict = dat.to_dict(orient='records')
vec = DictVectorizer()
# convert to array
matrix = vec.fit_transform(data_dict).toarray()

# apply kmeans
kmeans_ = kmeans.fit(matrix)

clusters = kmeans_.labels_
# don't change original dataset
dat["Clusters"] = clusters

############################### PCA prior to Visualization ####################

pca_2d = PCA(n_components=2)
PCs_2d = pd.DataFrame(pca_2d.fit_transform(dat.drop(["Clusters"], axis=1)))
PCs_2d.columns = ["PC1", "PC2"]
plotData = pd.concat([dat, PCs_2d], axis=1, join="inner")
x = SubClusterDF(k, plotData)
traces = SubScatterCluster(clusters=x, cols=colors, xlabel="PC1", ylabel="PC2")
fig = dict(data=traces, layout=layout)

st.plotly_chart(fig)

xlab = st.selectbox("For k-means, what would you like to plot on the x axis?", list(dat.columns))
ylab = st.selectbox("For k-means, what would you like to plot on the y axis?", list(dat.columns), index=5)
traces2 = SubScatterCluster(clusters=x, cols=colors, xlabel=xlab,  ylabel=ylab)
layout2 = dict(xaxis= dict(title=xlab, ticklen=5, zeroline=False),
               yaxis=dict(title=ylab, ticklen=5, zeroline=False))

fig2 = dict(data=traces2, layout=layout2)
st.plotly_chart(fig2)

############################### Cluster evaluation ############################
if len(np.unique(clusters)) > 1:
    silhouette = silhouette_score(dat, clusters)
    st.markdown("Silhouette Coefficient:")
    st.write(silhouette)

    davies_bouldin = davies_bouldin_score(dat, clusters)
    st.markdown("Davies Bouldin index:")
    st.write(davies_bouldin)
else:
    st.markdown("No evaluation can be calculated for one cluster")

###############################  DBSCAN ##########################################

st.markdown(
    """
    ## DBSCAN
    """
)


typ_dbscan = st.selectbox("Variables", ["numerical", "categorical", "both"],
                   help="to which type of variables DBSCAN should be applied")

dat_dbscan = dat_unmod.copy()



if typ_dbscan == "both":
    met_dbscan = st.selectbox("Distance Metric", ["gower"])
    st.markdown(
        """
        [__Gower's Distance__](https://medium.com/analytics-vidhya/gowers-distance-899f9c4bd553)
        is a metric which is used on mixed data.
        Gower's distance measure is computed as the average of partial dissimilarities across
        individuals. For quantitative/numerical data, the range normalized Manhatten distance is used,
        whereas the dice coefficient which is similar to the jaccard coefficient is used for categorical variables.
        """
    )
    dat_dbscan = PreProcessing(dat_dbscan, cat=True, mixed=True)  # only performs one-hot encoder

elif typ_dbscan == "categorical":
    met_dbscan = st.selectbox("Distance Metric", ["jaccard", "dice"])
    st.markdown(
        """
        The main [difference](https://stats.stackexchange.com/questions/55798/what-is-the-optimal-distance-function-for-individuals-when-attributes-are-nomina/55802#55802)
        between the jaccard and dice coefficient is that dice makes use of the fact that for former nominal
        variables which where converted to binary variables by one-hot encoding a match is as important as
        a mismatch. Therfore match and mismatch are assigned equal weights.
        """
    )
    dat_dbscan = PreProcessing(dat_dbscan, cat=True)

elif typ_dbscan == "numerical":
    met_dbscan = st.selectbox("Distance Metric", ["euclidean", "manhattan"])
    if met_dbscan == "euclidean":
        st.markdown(
            """
            For euclidean distance, numerical values will be standardized.
            """
        )
        dat_dbscan = PreProcessing(dat_dbscan, num=True)

    elif met_dbscan == "manhattan":
        dat_dbscan = dat_dbscan[list(dat_dbscan.columns[dat_dbscan.dtypes != 'category'])]




# connvert data to dictionary
data_dict_dbscan = dat_dbscan.to_dict(orient='records')
vec_dbscan = DictVectorizer()
# convert to array
matrix_dbscan = vec_dbscan.fit_transform(data_dict_dbscan).toarray()

# compute distance matrix
if met_dbscan == "gower":
    dist_matrix_dbscan = gower.gower_matrix(dat_dbscan)

else:
    dist_matrix_dbscan = pairwise_distances(matrix_dbscan, metric=met_dbscan)


min_samp_dbscan = st.slider('Minimum number of samples',
                     min_value=1,
                     max_value=30,
                     help="The number of samples in a neighborhood for a point to be considered as a core point")

if typ_dbscan == "both" or typ_dbscan == "categorical":
    radius = st.slider('Radius',
                       min_value=0.01,
                       max_value=1.0,
                       step=0.01,
                       help="from a core point, other datapoints are close if they lie within radius")
else:
    max_eps = float(abs(round(dist_matrix_dbscan.max())))
    radius = st.slider('Radius', min_value=1.0,
                       max_value=max_eps,
                       step=0.25,
                       help="from a core point, other datapoints are close if they lie within radius")
# create DBSCAN object
dbscan = DBSCAN(eps=radius, min_samples=min_samp_dbscan, metric="precomputed")

# apply DBSCAN
dbscan_ = dbscan.fit(dist_matrix_dbscan)
clusters_dbscan = dbscan_.labels_

dat_dbscan["Clusters"] = clusters_dbscan

############################### PCA prior to Visualization ####################

pca_2d_dbscan = PCA(n_components=2)
PCs_2d_dbscan = pd.DataFrame(pca_2d_dbscan.fit_transform(dat_dbscan.drop(["Clusters"], axis=1)))
PCs_2d_dbscan.columns = ["PC1", "PC2"]
plotData_dbscan = pd.concat([dat_dbscan, PCs_2d_dbscan], axis=1, join="inner")
x_dbscan = SubClusterDF(len(clusters_dbscan), plotData_dbscan)
traces_dbscan = SubScatterCluster(clusters=x_dbscan, cols=colors, xlabel="PC1", ylabel="PC2")
fig_dbscan = dict(data=traces_dbscan, layout=layout)

st.plotly_chart(fig_dbscan)

xlab_dbscan = st.selectbox("For DBSCAN, what would you like to plot on the x axis?", list(dat_dbscan.columns))
ylab_dbscan = st.selectbox("For DBSCAN, what would you like to plot on the y axis?", list(dat_dbscan.columns), index=5)
traces2_dbscan = SubScatterCluster(clusters=x_dbscan, cols=colors, xlabel=xlab_dbscan, ylabel=ylab_dbscan)
layout2_dbscan = dict(xaxis= dict(title=xlab_dbscan, ticklen=5, zeroline=False),
               yaxis=dict(title=ylab_dbscan, ticklen=5, zeroline=False))

fig2_dbscan = dict(data=traces2_dbscan, layout=layout2_dbscan)

st.plotly_chart(fig2_dbscan)

############################### Cluster evaluation ############################
if len(np.unique(clusters_dbscan)) > 1:
    silhouette = silhouette_score(dat_dbscan, clusters_dbscan)
    st.markdown("Silhouette Coefficient:")
    st.write(silhouette)

    davies_bouldin = davies_bouldin_score(dat_dbscan, clusters_dbscan)
    st.markdown("Davies Bouldin index:")
    st.write(davies_bouldin)
else:
    st.markdown("No evaluation can be calculated for one cluster")

############################### OPTICS ##########################################

st.markdown(
    """
    ## OPTICS
    """
)

typ_optics = st.selectbox("variables", ["numerical", "categorical", "both"],
                          help="to which type of variables DBSCAN should be applied")

dat_optics = dat_unmod.copy()

if typ_optics == "both":
    met_optics = st.selectbox("distance metric", ["gower"])

    dat_optics = PreProcessing(dat_optics, cat=True, mixed=True)  # only performs one-hot encoder

elif typ_optics == "categorical":
    met_optics = st.selectbox("distance metric", ["jaccard", "dice"])

    dat_optics = PreProcessing(dat_optics, cat=True)

elif typ_optics == "numerical":
    met_optics = st.selectbox("distance metric", ["euclidean", "manhattan"])
    if met_optics == "euclidean":
        dat_optics = PreProcessing(dat_optics, num=True)

    elif met_optics == "manhattan":
        dat_optics = dat_optics[list(dat_optics.columns[dat_optics.dtypes != 'category'])]



# connvert data to dictionary
data_dict_optics = dat_optics.to_dict(orient='records')
vec_optics = DictVectorizer()
# convert to array
matrix_optics = vec_optics.fit_transform(data_dict_optics).toarray()

# compute distance matrix
if met_optics == "gower":
    dist_matrix_optics = gower.gower_matrix(dat_optics)

else:
    dist_matrix_optics = pairwise_distances(matrix_optics, metric=met_optics)

min_samp = st.slider('minimum number of samples',
                     min_value=1,
                     max_value=30,
                     help="The number of samples in a neighborhood for a point to be considered as a core point")

optics = OPTICS(min_samples=min_samp, metric="precomputed")

# apply OPTICS
optics_ = optics.fit(dist_matrix_optics)
clusters_optics = optics_.labels_

dat_optics["Clusters"] = clusters_optics

############################### PCA prior to Visualization ####################

pca_2d_optics = PCA(n_components=2)
PCs_2d_optics = pd.DataFrame(pca_2d_optics.fit_transform(dat_optics.drop(["Clusters"], axis=1)))
PCs_2d_optics.columns = ["PC1", "PC2"]
plotData_optics = pd.concat([dat_optics, PCs_2d_optics], axis=1, join="inner")
x_optics = SubClusterDF(len(clusters_optics), plotData_optics)
traces_optics = SubScatterCluster(clusters=x_optics, cols=colors, xlabel="PC1", ylabel="PC2")
fig_optics = dict(data=traces_optics, layout=layout)

st.plotly_chart(fig_optics)

xlab_optics = st.selectbox("For OPTICS, what would you like to plot on the x axis?", list(dat_optics.columns))
ylab_optics = st.selectbox("For OPTICS, what would you like to plot on the y axis?", list(dat_optics.columns), index=5)
traces2_optics = SubScatterCluster(clusters=x_optics, cols=colors, xlabel=xlab_optics, ylabel=ylab_optics)
layout2_optics= dict(xaxis= dict(title=xlab_optics, ticklen=5, zeroline=False),
               yaxis=dict(title=ylab_optics, ticklen=5, zeroline=False))

fig2_optics = dict(data=traces2_optics, layout=layout2_optics)

st.plotly_chart(fig2_optics)
############################### Cluster evaluation ############################
if len(np.unique(clusters_optics)) > 1:
    silhouette = silhouette_score(dat_optics, clusters_optics)
    st.markdown("Silhouette Coefficient:")
    st.write(silhouette)

    davies_bouldin = davies_bouldin_score(dat_optics, clusters_optics)
    st.markdown("Davies Bouldin index:")
    st.write(davies_bouldin)
else:
    st.markdown("No internal evaluation can be calculated for a single cluster")