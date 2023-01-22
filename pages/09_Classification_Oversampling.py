import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

def calculate_metrics(y_test, y_pred):
    """Calculate f1 and auc and return them as rounded strings."""

    f1 = metrics.f1_score(y_test, y_pred)
    f1_str = str(round(100 * f1, 2))

    auc = metrics.roc_auc_score(y_test, y_pred)
    auc_str = str(round(auc, 2))

    return f1_str, auc_str

def create_decision_tree(X_train, y_train):
    # create inital decision tree
    decision_tree = tree.DecisionTreeClassifier(random_state=0)
    decision_tree = decision_tree.fit(X_train, y_train)

    # find optimal decision tree
    path = decision_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    accuray_max = 0
    opt_decision_tree = None

    for ccp_alpha in ccp_alphas:
        dt = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        scores = model_selection.cross_val_score(dt, X_train, y_train, cv=5)
        accuracy = np.mean(scores)
        if accuracy > accuray_max :
            accuray_max = accuracy
            opt_decision_tree = dt

    return opt_decision_tree.fit(X_train, y_train)

st.set_page_config(page_title="Classification - Oversampling")
st.markdown(
"""
# Working with imbalanced data
Unbalanced datasets are those in which an uneven distribution of classes, a so-called skewness
or bias, is found. One method to improve the distribution of training examples is to add or remove
examples in a process called resampling. Here, we will apply ovesampling methods which expand the partial
dataset of the minority class.
## Check class distribution in training dataset
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
With $$319$$ total counts of diagnosis "no heart disease" and $$371$$ total counts of diagnosis
"heart disease", class labels in our training dataset seem to be slightly shifted towards
a positive diagnosis of heart disease. However the skewness of this distribution does not seem severe.
Therefore we would not expect to seriously improve the performance of our decision-tree classifier by oversampling.
"""
)
st.markdown(
"""
## Apply Oversampling Methods
As the partial dataset with class labels "no hd" reflects the minority, we will add samples by oversampling.
### Decision Tree without Gender Feature
"""
)
ros = RandomOverSampler(random_state=2023)
X_tros, y_tros = ros.fit_resample(X_train, y_train)
X_smote, y_smote = SMOTE(random_state=2023).fit_resample(X_train, y_train)
# transform datasets to df
data_tros = pd.concat([X_tros, y_tros], axis=1)
data_test = pd.concat([X_test, y_test], axis=1)
data_ros = pd.concat([data_tros, data_test], axis=0)

data_tsmote = pd.concat([X_smote, y_smote], axis=1)
data_smote = pd.concat([data_tsmote, data_test], axis=0)

X_tros_without_sex = X_tros.drop(['sex'], axis=1)
X_test_without_sex = X_test.drop(['sex'], axis=1)

X_smote_without_sex = X_smote.drop(['sex'], axis=1)
# create basic decision tree without sex feature
decision_tree_without_sex_ros = create_decision_tree(X_train=X_tros_without_sex, y_train=y_tros)
decision_tree_without_sex_smote = create_decision_tree(X_train=X_smote_without_sex, y_train=y_tros)
# predict targets
y_ros_pred_without_sex = decision_tree_without_sex_ros.predict(X_test_without_sex)
y_smote_pred_without_sex = decision_tree_without_sex_smote.predict(X_test_without_sex)

meth_wosex = st.selectbox(
    label="Choose an oversampling method prior to generating decision-trees w/o sex information",
    options=["Random Oversampler", "SMOTE"],
    help="https://imbalanced-learn.org/stable/over_sampling.html"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
if meth_wosex == "Random Oversampler":
    st.write("A naive strategy to generate new samples is by randomly sampling with replacement.\
        Examples from the minority class are randomly selected and added to the training dataset.")
    tp = tree.plot_tree(decision_tree_without_sex_ros,
                        filled=True,
                        rounded=True,
                        class_names=['no_hd','yes_hd'],
                        feature_names=X_encode.columns)
    st.subheader("Confusion Matrix for Decision Tree without Gender Feature after random oversampling")
    cm_without_sex_ros = metrics.confusion_matrix(y_true=y_test, y_pred=y_ros_pred_without_sex)
    disp_ros = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_without_sex_ros)
    disp_ros.plot()
else:
    st.write("The **Synthetic Minority Oversampling Technique** (SMOTE) works by selecting examples\
             that are close in the feature space (neighbors), drawing a line between the examples in the feature\
            space and drawing a new sample at a point along that line.")
    tp = tree.plot_tree(decision_tree_without_sex_smote,
                        filled=True,
                        rounded=True,
                        class_names=['no_hd','yes_hd'],
                        feature_names=X_encode.columns)
    st.subheader("Confusion Matrix for Decision Tree without Gender Feature after SMOTE")
    cm_without_sex_smote = metrics.confusion_matrix(y_true=y_test, y_pred=y_smote_pred_without_sex)
    disp_smote = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_without_sex_smote)
    disp_smote.plot()

st.pyplot()

st.markdown(
"""Again, we evaluated the performance of the decision tree with a confusion matrix and by calculating
the area under the ROC curve (AUC) and the F-Score.
"""
)
f1 = 83.3
auc = 0.78
f1_ros, auc_ros = calculate_metrics(y_test, y_ros_pred_without_sex)
f1_smote, auc_smote = calculate_metrics(y_test, y_smote_pred_without_sex)

col1, col2, col3 = st.columns(3)
col1.write("🙅🏻‍♀️ No Oversampling")
with col1:
     st.metric(label="F-Score", value=f1)
     st.metric(label="AUC", value=auc)
col2.write("🎲 Random Oversampling")
with col2:
     st.metric(label="F-Score", value=f1_ros)
     st.metric(label="AUC", value=auc_ros)
col3.write("🏘 SMOTE")
with col3:
     st.metric(label="F-Score", value=f1_smote)
     st.metric(label="AUC", value=auc_smote)

st.markdown(
"""Again, we evaluated the performance of the decision tree with a confusion matrix and by calculating
the area under the ROC curve (AUC) and the F-Score.
"""
)