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
or bias, is found in the training data set for classification prediction modeling.
To check for skewness we'll plot the distribution of our training dataset based on class labels:
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
class_distr = training_dataset.groupby('sex')['hd'].value_counts()
#class_distr = class_distr.rename(index={'Female': 0, 'Male': 1})
#.unstack()
st.write(class_distr)

