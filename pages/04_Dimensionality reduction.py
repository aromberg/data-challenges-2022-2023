import sklearn as sk
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.set_page_config(page_title="Dimensionality Reduction",
                   page_icon="📌")

st.markdown("# Dimensionality Reduction")

### preparing data #########

all = pd.read_csv("./../data/data_imp.csv")

feature = st.selectbox("Select the feature by which to seperate the data",all.columns)
if feature != None:
    if feature in ["sex","num"]:
        cut_off = st.select_slider("select cut off",pd.unique(all[feature]))
    else:
        cut_off = st.slider("select the cut off",min(all[feature]),max(all[feature]))
    min_max = st.select_slider("Choose whether the before chosen value is minimum or maximum",["minimum","maximum"])
    if min_max == "minimum":
        all = all.loc[all[feature] >= cut_off]
    else:
        all = all.loc[all[feature] <= cut_off]
#######################
st.markdown(" ## PCA")
### PCA ###############

pca = PCA(n_components=7)
transformed_pca = pca.fit_transform(all)
transformed_pca = pd.DataFrame(transformed_pca)

# plot ###

x_achsis = st.selectbox("select the PC displayed on the x-achsis",transformed_pca.columns)
y_achsis = st.selectbox("select the PC displayed on the y-achsis",transformed_pca.columns)

figure = plt.figure()

plt.scatter((transformed_pca)[x_achsis], (transformed_pca)[y_achsis])


#plt.scatter((transformed_pca[transformed_pca.sex == 0])[x_achsis], (transformed_pca[transformed_pca.sex == 0])[y_achsis])
#plt.scatter((transformed_pca[transformed_pca.sex == 1])[x_achsis], (transformed_pca[transformed_pca.sex == 1])[y_achsis])
plt.xlabel("PC"+str(x_achsis))
plt.ylabel("PC"+str(y_achsis))
st.pyplot(figure)


#######################

st.markdown(" ## t-SNE")
### t-SNE ############

tsne = TSNE().fit_transform(all)
transformed_tsne = pd.DataFrame(tsne)

transformed_tsne["sex"] = all["sex"]

figure2 = plt.figure()
plt.scatter((transformed_tsne[transformed_tsne.sex == 0])[0], (transformed_tsne[transformed_tsne.sex == 0])[1])
plt.scatter((transformed_tsne[transformed_tsne.sex == 1])[0], (transformed_tsne[transformed_tsne.sex == 1])[1])

#plt.scatter(transformed_tsne[0],transformed_tsne[1])
st.pyplot(figure2)

######################

## UMAP ##############

#####################