import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Data Imputation",
                   page_icon="🎛️")

st.markdown("# Data Imputation")

st.markdown('We printed a preview of the dataset.')
# load dataset
data = pd.read_csv("data/heart_disease/heart_disease_combined.csv", sep=",",
 index_col=0)
st.dataframe(data.head())

st.markdown('We calculated the means for each continuous variable for both \
 male and female to identify differences in the data. We can see that data \
 of variable "trestbps" seems to be similiar for males and females. That \
 is why we deleted this column.')

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

st.markdown('## Mean Imputation for "chol"')
st.markdown('The blue datapoints represent the complete datapoints and \
and the orange datapoints represent the datapoints where we imputet the mean.')
data_imp['chol'] = data_imp['chol'].fillna(data_imp['chol'].mean())
# visualize results
fig = plt.figure()
plt.scatter(data_imp['age'], data_imp['chol'])
plt.scatter(data_imp[np.isnan(data['chol'])]['age'],
 data_imp[np.isnan(data['chol'])]['chol'])
plt.xlabel('age')
plt.ylabel('chol')
st.pyplot(fig)

st.markdown('## Mean Imputation for "thalach"')
data_imp['thalach'] = data_imp['thalach'].fillna(data_imp['thalach'].mean())
# visualize results
fig = plt.figure()
plt.scatter(data_imp['age'], data_imp['thalach'])
plt.scatter(data_imp[np.isnan(data['thalach'])]['age'],
 data_imp[np.isnan(data['thalach'])]['thalach'])
plt.xlabel('age')
plt.ylabel('thalach')
st.pyplot(fig)

st.markdown('## Mean Imputation for "oldpeak"')
data_imp['oldpeak'] = data_imp['oldpeak'].fillna(data_imp['oldpeak'].mean())
# visualize results
fig = plt.figure()
plt.scatter(data_imp['age'], data_imp['oldpeak'])
plt.scatter(data_imp[np.isnan(data['oldpeak'])]['age'],
 data_imp[np.isnan(data['oldpeak'])]['oldpeak'])
plt.xlabel('age')
plt.ylabel('oldpeak')
st.pyplot(fig)

st.markdown('Analogous to the continuous variable we calculated the modes for \
each categorial variable for both male and female and delete all columns \
which indicate no difference.')
modes_male = data_male.mode().iloc[0]
modes_female = data_female.mode().iloc[0]

modes = pd.DataFrame([modes_male, modes_female])
modes.index = ['male', 'female']
modes = modes.drop(['age','sex','trestbps','chol','thalach','oldpeak','num'],
 axis=1)
st.dataframe(modes)

# delete variables where is no difference in male and female
data_imp = data_imp.drop(columns=['cp','fbs','restecg','exang','slope','ca'])

st.markdown('## Mode Imputation for "thal"')
# mode imputation
data_imp['thal'] = data_imp['thal'].fillna(int(data_imp['thal'].mode()))
null_values = data['thal'].isnull()

# visualize results
fig = plt.figure()
plt.scatter(data_imp['age'], data_imp['thal'])
plt.scatter(data_imp[np.isnan(data['thal'])]['age'],
 data_imp[np.isnan(data['thal'])]['thal'])
plt.xlabel('age')
plt.ylabel('thal')
st.pyplot(fig)

st.markdown('## Mean Imputation for subgroups')
st.markdown('To check how the imputation values differ if we apply them to \
subgroups we divided the dataset in males and females.')
data_imp_male = data[data['sex']==1].copy()
data_imp_female = data[data['sex']==0].copy()

data_imp_male['chol'] = \
data_imp_male['chol'].fillna(data_imp_male['chol'].mean())
data_imp_female['chol'] = \
data_imp_female['chol'].fillna(data_imp_female['chol'].mean())

data_imp_male['thalach'] = \
data_imp_male['thalach'].fillna(data_imp_male['thalach'].mean())
data_imp_female['thalach'] = \
data_imp_female['thalach'].fillna(data_imp_female['thalach'].mean())

data_imp_male['oldpeak'] = \
data_imp_male['oldpeak'].fillna(data_imp_male['oldpeak'].mean())
data_imp_female['oldpeak'] = \
data_imp_female['oldpeak'].fillna(data_imp_female['oldpeak'].mean())

data_imp_male['thal'] = \
data_imp_male['thal'].fillna(data_imp_male['thal'].mode())
data_imp_female['thal'] = \
data_imp_female['thal'].fillna(data_imp_female['thal'].mode())

# visualize results
fig = plt.figure()
plt.scatter(data_imp_male['age'], data_imp_male['oldpeak'])
plt.scatter(data_imp_male[np.isnan(data[data['sex']==1]. \
copy()['oldpeak'])]['age'],data_imp_male[ \
np.isnan(data[data['sex']==1].copy()['oldpeak'])]['oldpeak'])
plt.xlabel('age')
plt.ylabel('oldpeak')
plt.title('Mean Imputation for Males')
st.pyplot(fig)

fig = plt.figure()
plt.scatter(data_imp_female['age'], data_imp_female['oldpeak'])
plt.scatter(data_imp_female[np.isnan(data[data['sex']==0]['oldpeak'])]['age'],
    data_imp_female[np.isnan(data[data['sex']==0]['oldpeak'])]['oldpeak'])
plt.xlabel('age')
plt.ylabel('oldpeak')
plt.title('Mean Imputation for Females')
st.pyplot(fig)

# concat male and female datasets
data_imp = pd.concat([data_imp_male, data_imp_female]).sort_index()
fig = plt.figure()
plt.scatter(data_imp['age'], data_imp['oldpeak'])
plt.scatter(data_imp[np.isnan(data['oldpeak'])]['age'],
    data_imp[np.isnan(data['oldpeak'])]['oldpeak'])
plt.xlabel('age')
plt.ylabel('oldpeak')
plt.title('Mean Imputation for Males and Females')
st.pyplot(fig)

# save results
from pathlib import Path
filepath = Path('data/data_imp.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
data_imp.to_csv(filepath, index=False)
