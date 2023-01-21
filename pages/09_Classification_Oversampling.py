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
To check for skewness we'll plot the distribution of our training dataset based on class labels and gender
and calculate the sample skewness:
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
# compute skewness
#biased = skew(class_distr.to_numpy(), axis=1)
biased = class_distr.skew(axis = 0, skipna = True)
# plot
# set height
female = class_distr.transpose()['Female'].values.tolist()
male = class_distr.transpose()['Male'].values.tolist()
x = np.arange(len(female))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x, female, color = 'darkviolet', width = 0.25, label ='Female')
ax.bar(x, male, color = 'darkorange', width = 0.25, label ='Male', bottom=female)
# look
plt.title('Class Distribution')
plt.xlabel('hd', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks(x, list(class_distr.columns))
plt.legend(labels=list(class_distr.transpose().columns))
st.pyplot(fig)


st.markdown(
"""
The overall distribution of class labels seem to be slightly shifted towards
a positive diagnosis of heart disease which is confirmed by the Fisher-Pearson coefficient of skewness
of {}.
""".format(biased)
)
