import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
from scipy.stats import skew
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Classification - Oversampling")
st.markdown(
"""
# Working with imbalanced data
## Check class distribution in training dataset
Unbalanced datasets are those in which an uneven distribution of classes, a so-called skewness
or bias, is found.
To check for skewness we'll plot the distribution of our training dataset based on class labels and gender:
""")

data = pd.read_csv("data/data_hr.csv")
# split features and targets
X = data.iloc[:,:13].copy()
y = data['hd'].copy()

# encode categorial values
X_encode = pd.get_dummies(X, columns=['cp','restecg','slope','thal'])

# split training and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_encode, y, random_state=1, test_size=0.25)

# delete sex feature
data_without_sex = data.drop(['sex'], axis=1)
X_train_without_sex = X_train.drop(['sex'], axis=1)
X_test_without_sex = X_test.drop(['sex'], axis=1)

training_dataset = pd.concat([X_train, y_train], axis=1)
class_distr = training_dataset.groupby('sex')['hd'].value_counts().unstack()
class_distr = class_distr.rename(index={0: 'Female', 1: 'Male'}, columns={0: 'No', 1: 'Yes'})
class_distr
# plot
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
# set height
female = class_distr.transpose()['Female'].values.tolist()
male = class_distr.transpose()['Male'].values.tolist()
# pos of bar
br1 = np.arange(len(female))
br2 = [x + barWidth for x in br1]
# make plot
plt.bar(br1, female, color ='r', width = barWidth,
        edgecolor ='grey', label ='Female')
plt.bar(br2, male, color ='g', width = barWidth,
        edgecolor ='grey', label ='Male')
# look
plt.xlabel('hd', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(female))],
        list(class_distr.columns))
 
plt.legend()
st.pyplot(fig)