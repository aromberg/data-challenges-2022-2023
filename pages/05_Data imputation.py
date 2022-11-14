import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Data Imputation",
                   page_icon="🎛️")

st.markdown("# Data Imputation")

st.markdown('dataset')
# load dataset
data = pd.read_csv("data/heart_disease/heart_disease_combined.csv", sep=",",
 index_col=0)
st.dataframe(data.head())

st.markdown('calculate means for each continuous variable for both male and female')
data_male = data[data['sex'] == 1]
data_female = data[data['sex'] == 0]

means_male = data_male.mean()
means_female = data_female.mean()

means = pd.DataFrame([means_male, means_female])
means.index = ['male', 'female']
means = means.drop(['age','sex','cp','fbs', 'restecg','exang','slope','thal',
'num','ca'], axis=1)
st.dataframe(means)

# delete variables where is no difference in male and female
data_imp = data.copy()
data_imp = data_imp.drop(columns=['trestbps'])

st.markdown('mean imputation for chol')
data_imp['chol'] = data_imp['chol'].fillna(data_imp['chol'].mean())
# visualize results
fig = plt.figure()
plt.scatter(data_imp['age'], data_imp['chol'])
plt.scatter(data_imp[np.isnan(data['chol'])]['age'],
 data_imp[np.isnan(data['chol'])]['chol'])
plt.xlabel('age')
plt.ylabel('chol')
st.pyplot(fig)

st.markdown('mean imputation for thalach')
data_imp['thalach'] = data_imp['thalach'].fillna(data_imp['thalach'].mean())
# visualize results
fig = plt.figure()
plt.scatter(data_imp['age'], data_imp['thalach'])
plt.scatter(data_imp[np.isnan(data['thalach'])]['age'],
 data_imp[np.isnan(data['thalach'])]['thalach'])
plt.xlabel('age')
plt.ylabel('thalach')
st.pyplot(fig)

st.markdown('mean imputation for oldpeak')
data_imp['oldpeak'] = data_imp['oldpeak'].fillna(data_imp['oldpeak'].mean())
# visualize results
fig = plt.figure()
plt.scatter(data_imp['age'], data_imp['oldpeak'])
plt.scatter(data_imp[np.isnan(data['oldpeak'])]['age'],
 data_imp[np.isnan(data['oldpeak'])]['oldpeak'])
plt.xlabel('age')
plt.ylabel('oldpeak')
st.pyplot(fig)

st.markdown('calculate modes for each categorial variable for both male and female')
modes_male = data_male.mode().iloc[0]
modes_female = data_female.mode().iloc[0]

modes = pd.DataFrame([modes_male, modes_female])
modes.index = ['male', 'female']
modes = modes.drop(['age','sex','trestbps','chol','thalach','oldpeak','num'], axis=1)
st.dataframe(modes)

# delete variables where is no difference in male and female
data_imp = data_imp.drop(columns=['cp','fbs','restecg','exang','slope','ca'])


# mode imputation
data_imp['thal'] = data_imp['thal'].fillna(int(data_imp['thal'].mode()))
null_values = data['thal'].isnull()

# save results
from pathlib import Path
filepath = Path('../data/data_imp.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
data_imp.to_csv(filepath, index=False)

# visualize results
fig = plt.figure()
plt.scatter(data_imp['age'], data_imp['thal'])
plt.scatter(data_imp[np.isnan(data['thal'])]['age'],
 data_imp[np.isnan(data['thal'])]['thal'])
plt.xlabel('age')
plt.ylabel('thal')
st.pyplot(fig)

st.markdown('mean imputation for subgroups divided in male and female')
data_imp_male = data[data['sex']==1].copy()
data_imp_female = data[data['sex']==0].copy()

data_imp_male['oldpeak'] = data_imp_male['oldpeak'].fillna(data_imp_male['oldpeak'].mean())
data_imp_female['oldpeak'] = data_imp_female['oldpeak'].fillna(data_imp_female['oldpeak'].mean())

# visualize results
fig = plt.figure()
plt.scatter(data_imp_male['age'], data_imp_male['oldpeak'])
plt.scatter(data_imp_male[np.isnan(data[data['sex']==1].copy()['oldpeak'])]['age'],
                   data_imp_male[np.isnan(data[data['sex']==1].copy()['oldpeak'])]['oldpeak'])
plt.xlabel('age')
plt.ylabel('oldpeak')
st.pyplot(fig)

fig = plt.figure()
plt.scatter(data_imp_female['age'], data_imp_female['oldpeak'])
plt.scatter(data_imp_female[np.isnan(data[data['sex']==0]['oldpeak'])]['age'],
                   data_imp_female[np.isnan(data[data['sex']==0]['oldpeak'])]['oldpeak'])
plt.xlabel('age')
plt.ylabel('oldpeak')
st.pyplot(fig)
