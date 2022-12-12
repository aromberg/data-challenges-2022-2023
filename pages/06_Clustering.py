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


def plot_clusters(df, n_clusters, xlab, ylab):
    """
    Plot clusters as PCA & column-based

    Args:
        df: data to plot
        name: cluster method name for widget inputs
        n_clusters: number of clusters

    Returns:
        None
    """
    pca_2d = PCA(n_components=2)
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(df.drop(["Clusters"], axis=1)))
    PCs_2d.columns = ["PC1", "PC2"]
    plotData = pd.concat([df, PCs_2d], axis=1, join="inner")
    x = SubClusterDF(n_clusters, plotData)
    traces = SubScatterCluster(clusters=x, cols=colors, xlabel="PC1", ylabel="PC2")
    layout = dict(title="Cluster PCA Visualization",
                xaxis= dict(title="PC1", ticklen=5, zeroline=False),
                yaxis=dict(title="PC2", ticklen=5, zeroline=False))
    fig = dict(data=traces, layout=layout)

    st.plotly_chart(fig, use_container_width=True)

    traces2 = SubScatterCluster(clusters=x, cols=colors, xlabel=xlab,  ylabel=ylab)
    layout2 = dict(xaxis= dict(title=xlab, ticklen=5, zeroline=False),
                yaxis=dict(title=ylab, ticklen=5, zeroline=False))
    fig2 = dict(data=traces2, layout=layout2)
    st.plotly_chart(fig2, use_container_width=True)


def convert_data(df):
    """
    Convert data to array for analysis.

    Args:
        df: pandas dataframe to convert

    Returns:
        data as array
    """

    # convert data to dictionary
    data_dict = df.to_dict(orient='records')
    vec = DictVectorizer()
    # convert to array
    return vec.fit_transform(data_dict).toarray()

def evaluate_clusters(raw_data, clusters):
    if len(np.unique(clusters)) > 1:
        silhouette = silhouette_score(raw_data, clusters)
        st.markdown("Silhouette Coefficient:")
        st.write(silhouette)

        davies_bouldin = davies_bouldin_score(raw_data, clusters)
        st.markdown("Davies Bouldin index:")
        st.write(davies_bouldin)
    else:
        st.markdown("No evaluation can be calculated for one cluster")

def get_plot_labels(name, df):
    xlab = st.selectbox(f"For {name}, what would you like to plot on the x axis?", list(df.columns))
    ylab = st.selectbox(f"For {name}, what would you like to plot on the y axis?", list(df.columns), index=5)

    return xlab, ylab


####################### Prepare Data ##########################################

# read files
data = pd.read_csv("data/heart_disease/heart_disease_combined.csv")
data_imp = pd.read_csv("data/data_imp.csv")

# drop coll Unnamed: 0 and rows with NAs for raw data set
data = data.drop(labels="Unnamed: 0", axis=1)
data = data.dropna(axis=0)

# assign new dtypes
dtypes = {"sex": "category",
        "cp": "category",
        "restecg": "category",
        "fbs": "category",
        "exang": "category",
        "slope": "category",
        "ca": "category",
        "thal": "category",
        "num": "category"}

data = data.astype(dtypes)
data_imp = data_imp.astype(dtypes)

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


############################### Set Page ##########################################

###############################  K-Means ##########################################

st.set_page_config(page_title="Clustering",
                   page_icon="🗺")

st.markdown(
    """
    # Clustering 🗺
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
    ## Cluster evaluation
    ### Shilouette index
    The shilouette index is one way to calculate the relationship between the intra cluster similarity and the inter cluster similarity. For more then 2 clusters the average is calculated. 
    $$ I_s = \frac{d_inter - d_intra}{max(d_inter,d_intra)}$$
    The values ranges from 1 to -1, where 1 means, that the inter cluster distance is large, while the intra cluster distance is low. 0 means, 
    that there is no significant distance between the clusters and -1 means, that the clusters are assigned wrongly.
    ### Davies Bouldin index
    The Davies Bouldin index also represents the relationship between intra cluster similiarity and inter cluster similarity. This index is based around the cluster centroids. Intrasimilarity is 
    the average distance from the centroid within the cluster and the inter cluster similarity is given by the distance between the cluster centroids. 
    The similarity of two clusters is calculated as the sum of the two intra cluster distance divided by the inter cluster distance. The average over all cluster similarities gives the index value.
    Better distinguished clusters resul in a lower index value.
    """
)

# inputs
k = st.slider('Number of Clusters', value=3, min_value=1, max_value=8)
init_method = st.selectbox("Initialization Method", ["k-means++", "random"],
                           help="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html")
kmeans_type = st.selectbox("Variables", ["numerical", "numerical and categorical"],
                   help="If 'numerical and categorical' is choosen, numerical variables will be standardized and categorical features will be encoded as one-hot numeric array")

if kmeans_type == "numerical":
    stand = st.checkbox("Standardize features", value=True)
if init_method == "k-means++":
    kmeans = KMeans(n_clusters=k, init=init_method)
else:
    kmeans = KMeans(n_clusters=k, init=init_method, random_state=222)

# preprocess data
if kmeans_type == "numerical and categorical":
    data_kmeans = PreProcessing(data=data, num=True, cat=True, mixed=True)
    data_kmeans_imp = PreProcessing(data=data_imp, num=True, cat=True, mixed=True)
else:
    if stand:
        data_kmeans = PreProcessing(data=data, num=True)
        data_kmeans_imp = PreProcessing(data=data_imp, num=True)
    else:
        data_kmeans = data
        data_kmeans_imp = data_imp

# apply kmeans
kmeans_ = kmeans.fit(convert_data(data_kmeans))
clusters = kmeans_.labels_
data_kmeans["Clusters"] = clusters

kmeans_ = kmeans.fit(convert_data(data_kmeans_imp))
clusters_imp = kmeans_.labels_
data_kmeans_imp["Clusters"] = clusters_imp

# output
xlab, ylab = get_plot_labels("k-means", data_kmeans)

col1, col2 = st.columns(2)

with col1:
    plot_clusters(data_kmeans, k, xlab, ylab)
    evaluate_clusters(data_kmeans, clusters)
with col2:
    plot_clusters(data_kmeans_imp, k, xlab, ylab)
    evaluate_clusters(data_kmeans_imp, clusters_imp)

###############################  DBSCAN ##########################################

st.markdown(
    """
    ## DBSCAN
    """
)

data_dbscan = data.copy()
data_dbscan_imp = data_imp.copy()

# inputs & preprocessing
dbscan_type = st.selectbox("Variables", ["numerical", "categorical", "both"],
                   help="to which type of variables DBSCAN should be applied")

if dbscan_type == "both":
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
    data_dbscan = PreProcessing(data_dbscan, cat=True, mixed=True)  # only perform one-hot encoder
    data_dbscan_imp = PreProcessing(data_dbscan_imp, cat=True, mixed=True) # same here

elif dbscan_type == "categorical":
    met_dbscan = st.selectbox("Distance Metric", ["jaccard", "dice"])
    st.markdown(
        """
        The main [difference](https://stats.stackexchange.com/questions/55798/what-is-the-optimal-distance-function-for-individuals-when-attributes-are-nomina/55802#55802)
        between the jaccard and dice coefficient is that dice makes use of the fact that for former nominal
        variables which where converted to binary variables by one-hot encoding a match is as important as
        a mismatch. Therfore match and mismatch are assigned equal weights.
        """
    )
    data_dbscan = PreProcessing(data_dbscan, cat=True)
    data_dbscan_imp = PreProcessing(data_dbscan_imp, cat=True)

elif dbscan_type == "numerical":
    met_dbscan = st.selectbox("Distance Metric", ["euclidean", "manhattan"])
    if met_dbscan == "euclidean":
        st.markdown(
            """
            For euclidean distance, numerical values will be standardized.
            """
        )
        data_dbscan = PreProcessing(data_dbscan, num=True)
        data_dbscan_imp = PreProcessing(data_dbscan_imp, num=True)
    elif met_dbscan == "manhattan":
        data_dbscan = data_dbscan[list(data_dbscan.columns[data_dbscan.dtypes != 'category'])]
        data_dbscan_imp = data_dbscan_imp[list(data_dbscan_imp.columns[data_dbscan_imp.dtypes != 'category'])]

# compute distance matrices
if met_dbscan == "gower":
    dist_matrix_dbscan = gower.gower_matrix(data_dbscan)
    dist_matrix_dbscan_imp = gower.gower_matrix(data_dbscan_imp)
else:
    dist_matrix_dbscan = pairwise_distances(convert_data(data_dbscan), metric=met_dbscan)
    dist_matrix_dbscan_imp = pairwise_distances(convert_data(data_dbscan_imp), metric=met_dbscan)


# further inputs (dependend on distance matrix)
min_samp_dbscan = st.slider('Minimum number of samples',
                     min_value=1,
                     max_value=30,
                     help="The number of samples in a neighborhood for a point to be considered as a core point")
if dbscan_type == "both" or dbscan_type == "categorical":
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

# apply DBSCAN
dbscan = DBSCAN(eps=radius, min_samples=min_samp_dbscan, metric="precomputed")
dbscan_ = dbscan.fit(dist_matrix_dbscan)
clusters_dbscan = dbscan_.labels_
data_dbscan["Clusters"] = clusters_dbscan

dbscan_ = dbscan.fit(dist_matrix_dbscan_imp)
clusters_dbscan_imp = dbscan_.labels_
data_dbscan_imp["Clusters"] = clusters_dbscan_imp


# output
xlab, ylab = get_plot_labels("DBSCAN", data_dbscan)

col1, col2 = st.columns(2)

with col1:
    plot_clusters(data_dbscan, len(clusters_dbscan), xlab, ylab)
    evaluate_clusters(data_dbscan, clusters_dbscan)
with col2:
    plot_clusters(data_dbscan_imp, len(clusters_dbscan), xlab, ylab)
    evaluate_clusters(data_dbscan_imp, clusters_dbscan_imp)

############################### OPTICS ##########################################

st.markdown(
    """
    ## OPTICS
    """
)

optics_type = st.selectbox("variables", ["numerical", "categorical", "both"],
                          help="to which type of variables DBSCAN should be applied")

data_optics = data.copy()
data_optics_imp = data_imp.copy()

# inputs & preprocessing
if optics_type == "both":
    met_optics = st.selectbox("distance metric", ["gower"])

    data_optics = PreProcessing(data_optics, cat=True, mixed=True)  # only perform one-hot encoder
    data_optics_imp = PreProcessing(data_optics_imp, cat=True, mixed=True)  # same here!
elif optics_type == "categorical":
    met_optics = st.selectbox("distance metric", ["jaccard", "dice"])

    data_optics = PreProcessing(data_optics, cat=True)
    data_optics_imp = PreProcessing(data_optics_imp, cat=True)
elif optics_type == "numerical":
    met_optics = st.selectbox("distance metric", ["euclidean", "manhattan"])
    if met_optics == "euclidean":
        data_optics = PreProcessing(data_optics, num=True)
        data_optics_imp = PreProcessing(data_optics_imp, num=True)
        
    elif met_optics == "manhattan":
        data_optics = data_optics[list(data_optics.columns[data_optics.dtypes != 'category'])]
        data_optics_imp = data_optics_imp[list(data_optics_imp.columns[data_optics_imp.dtypes != 'category'])]

# compute distance matrix
if met_optics == "gower":
    dist_matrix_optics = gower.gower_matrix(data_optics)
    dist_matrix_optics_imp = gower.gower_matrix(data_optics_imp)

else:
    dist_matrix_optics = pairwise_distances(convert_data(data_optics), metric=met_optics)
    dist_matrix_optics_imp = pairwise_distances(convert_data(data_optics_imp), metric=met_optics)

min_samp = st.slider('minimum number of samples',
                     min_value=1,
                     max_value=30,
                     help="The number of samples in a neighborhood for a point to be considered as a core point")


# apply OPTICS
optics = OPTICS(min_samples=min_samp, metric="precomputed")

optics_ = optics.fit(dist_matrix_optics)
clusters_optics = optics_.labels_
data_optics["Clusters"] = clusters_optics

optics_ = optics.fit(dist_matrix_optics_imp)
clusters_optics_imp = optics_.labels_
data_optics_imp["Clusters"] = clusters_optics_imp

# output
xlab, ylab = get_plot_labels("OPTICS", data_optics)

col1, col2 = st.columns(2)

with col1:
    plot_clusters(data_optics, len(clusters_optics), xlab, ylab)
    evaluate_clusters(data_optics, clusters_optics)
with col2:
    plot_clusters(data_optics_imp, len(clusters_optics_imp), xlab, ylab)
    evaluate_clusters(data_optics_imp, clusters_optics_imp)