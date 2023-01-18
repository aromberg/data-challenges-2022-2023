import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt

st.set_page_config(page_title="Classification")

st.markdown(
"""
# Classification for Detecting Heart Disease
## Preprocessing Data
In order not to confuse numerical and categorical features, we made the dataset
a bit more human-readable. Note that we still use the imputed dataset.
""")
data = pd.read_csv("../data/data_hr.csv")
st.dataframe(data.head())

def create_decision_tree(data, X_train, y_train, X_test, y_test):
    # create inital decision tree
    decision_tree = tree.DecisionTreeClassifier(random_state=0)
    decision_tree = decision_tree.fit(X_train, y_train)

    # find optimal decision tree
    path = decision_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    accuray_max = 0
    ccp_alpha_max = ccp_alphas[0]
    opt_decision_tree = None

    for ccp_alpha in ccp_alphas:
        dt = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)
        if score > accuray_max :
            accuray_max = score
            ccp_alpha_max = ccp_alpha
            opt_decision_tree = dt

    return opt_decision_tree

st.markdown(
"""
## Decision Tree without Gender Feature
First, we create a decision tree using scikit-learn without considering gender \
as a criterion for decision making.
"""
)

# split features and targets
X = data.iloc[:,:13].copy()
y = data['hd'].copy()

# encode categorial values
X_encode = pd.get_dummies(X, columns=['cp','restecg','slope','thal'])

# split training and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_encode, y, random_state=1)

# delete sex feature
data_without_sex = data.drop(['sex'], axis=1)
X_train_without_sex = X_train.drop(['sex'], axis=1)
X_test_without_sex = X_test.drop(['sex'], axis=1)

# create basic decision tree without sex feature
decision_tree_without_sex = create_decision_tree(data_without_sex, X_train_without_sex, y_train, X_test_without_sex, y_test)

tp = tree.plot_tree(decision_tree_without_sex,
              filled=True,
              rounded=True,
              class_names=['no_hd','yes_hd'],
              feature_names=X_encode.columns)
st.pyplot()

st.subheader("Confusion Matrix for Decision Tree without Gender Feature")
metrics.plot_confusion_matrix(decision_tree_without_sex, X_test_without_sex, y_test, display_labels=['Does not have HD','Does have HD'])
st.pyplot()

st.markdown(
"We evaluated the performance of the decision tree with a confusion matrix."
)

counts_hd_no = len(data[data['hd']==0])
counts_hd_yes = len(data[data['hd']==1])

y_pred = decision_tree_without_sex.predict(X_test_without_sex)
score = metrics.accuracy_score(y_test, y_pred)

st.markdown(
"We check the balance of the dataset with respect to heart disease. \
The number of records without heard disease is " + str(counts_hd_no) +
" and the number of records with heard disease is " + str(counts_hd_yes) + ". \
There is no large imbalance, which means we can use the accuary as a \
performance measure. The accuray is " + str(100 * round(score, 4)) + "%."
)

################################################################################

st.markdown(
"""
## Basic Decision Tree
Now we take gender into account to see if it improves the classifier.
""")

# create basic decision tree
decision_tree = create_decision_tree(data, X_train, y_train, X_test, y_test)

# visualize decision tree
#st.set_option('deprecation.showPyplotGlobalUse', False)
tp = tree.plot_tree(decision_tree,
              filled=True,
              rounded=True,
              class_names=['no_hd','yes_hd'],
              feature_names=X_encode.columns)
st.pyplot()

st.markdown(
"If we look at the decision criteria in the tree, we see that gender occurs once."
)

# calculate confusion matrix
st.set_option('deprecation.showPyplotGlobalUse', False)
st.subheader("Confusion Matrix for Basic Decision Tree")
metrics.plot_confusion_matrix(decision_tree, X_test, y_test, display_labels=['Does not have HD','Does have HD'])
st.pyplot()

y_pred = decision_tree.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)

st.markdown(
"The accuray is " + str(100 * round(score, 4)) + "%. So we have been able to \
improve performance. Gender seems to be a medical criterion for predicting the \
disease."
)
