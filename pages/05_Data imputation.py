import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Data Imputation",
                   page_icon="🎛️")

st.markdown("# Data Imputation")

st.markdown('Printing a preview of the datasets reveals missing data:')

# load dataset
data = pd.read_csv("data/heart_disease/heart_disease_combined.csv", sep=",",
 index_col=0)
st.dataframe(data.head())

# lists for discrete/continuous data
cont_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
disc_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']

st.markdown(
    """
    ## Comparing male and female NaNs
    Because this project is about finding differnces between genders, we \
    decided to see if differences already exists and this stage, and if \
    imputation should thus be done separately:

    ### Histograms
    """
)
# plot attributes to get a first impression of them before imputing
compare_option = st.selectbox(
    'Which column would you like to plot?',
    data.columns)

col1, col2 = st.columns(2)

with col1:
    st.write("#### Male")
    hist = px.histogram(data[data['sex'] == 1], x=compare_option)
    st.plotly_chart(hist, use_container_width=True)

with col2:
    st.write("#### Female")
    hist = px.histogram(data[data['sex'] == 0], x=compare_option)
    st.plotly_chart(hist, use_container_width=True)


st.write("### Means")
st.markdown('We calculated the means for each continuous variable for both \
 male and female to identify differences in the data. We can see that data \
 of variable "trestbps" seems to be similiar for males and females.')
means_male = data[data['sex'] == 1].mean()
means_female = data[data['sex'] == 0].mean()

means = pd.DataFrame([means_male, means_female])
means.index = ['male', 'female']
means = means.drop(disc_columns, axis=1)
st.write(means)


st.write("### Modes")
st.markdown('Analogous to the continuous variable we calculated the modes for \
each categorial variable for both male and female and delete all columns \
which indicate no difference.')
modes_male = data[data['sex'] == 1].mode().iloc[0]
modes_female = data[data['sex'] == 0].mode().iloc[0]

modes = pd.DataFrame([modes_male, modes_female])
modes.index = ['male', 'female']
modes = modes.drop(cont_columns, axis=1)
st.write(modes)

# Imputation
data_imp_male = data.copy()[data['sex'] == 1]
data_imp_female = data.copy()[data['sex'] == 0]


st.write("## Mean & mode imputation")
st.markdown('Because significant differences are visible when comparing male\
             and female data, we decided to impute missing values separately.')

# mean imputation
male_imp_dict = means.to_dict(orient='records')[0]
female_imp_dict = means.to_dict(orient='records')[1]
data_imp_male.fillna(male_imp_dict, inplace=True)
data_imp_female.fillna(female_imp_dict, inplace=True)
# st.write(data_imp_female)

# mode imputation
male_imp_dict = modes.to_dict(orient='records')[0]
female_imp_dict = modes.to_dict(orient='records')[1]
data_imp_male.fillna(male_imp_dict, inplace=True)
data_imp_female.fillna(female_imp_dict, inplace=True)

data_imp = pd.concat([data_imp_male, data_imp_female]).sort_index()

st.write('## Results')

# dropdown for chart
imputed_option = st.selectbox(
    'Which column would you like to plot to see imputations?',
    data_imp.columns,
    index=4)

fig = plt.figure()
plt.scatter(data_imp['age'], data_imp[imputed_option])
plt.scatter(data_imp[np.isnan(data[imputed_option])]['age'],
 data_imp[np.isnan(data[imputed_option])][imputed_option])
plt.xlabel('age')
plt.ylabel(imputed_option)
st.pyplot(fig)

with st.expander("See imputed dataset"):
    st.write(data_imp)


# save results
from pathlib import Path
filepath = Path('data/data_imp.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
data_imp.to_csv(filepath, index=False)