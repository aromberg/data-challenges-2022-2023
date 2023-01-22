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
decision_tree_without_sex_smote = create_decision_tree(X_train=X_smote_without_sex, y_train=y_smote)
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
    st.pyplot()                
    cm_without_sex_ros = metrics.confusion_matrix(y_true=y_test, y_pred=y_ros_pred_without_sex)
    disp_ros = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_without_sex_ros)
    disp_ros.plot()
    st.pyplot()
else:
    st.write("The **Synthetic Minority Oversampling Technique** (SMOTE) works by selecting examples\
             that are close in the feature space (neighbors), drawing a line between the examples in the feature\
            space and drawing a new sample at a point along that line.")
    tp = tree.plot_tree(decision_tree_without_sex_smote,
                        filled=True,
                        rounded=True,
                        class_names=['no_hd','yes_hd'],
                        feature_names=X_encode.columns)
    st.pyplot()
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

col1, col2, col3 = st.columns(3, gap="medium")
col1.write("🙅🏻‍♀️ No Oversampling")
with col1:
     st.metric(label="F-Score", value=f"{f1}%")
     st.metric(label="AUC", value=auc)
col2.write("🎲 Random Oversampling")
with col2:
     st.metric(label="F-Score", value=f"{f1_ros}%")
     st.metric(label="AUC", value=auc_ros)
col3.write("🏘 SMOTE")
with col3:
     st.metric(label="F-Score", value=f"{f1_smote}%")
     st.metric(label="AUC", value=auc_smote)


st.markdown(
"""With respect to the F-Score and AUC of the decision tree calculated when there was no oversampling
    applied, model performance seems to have reduced when using the random oversampling method but improved
    when using SMOTE. This could be explained by a potential overfitting of our model when using random
    oversampling. However, the scores are close to each other, not resulting in a severe improvement
    of our model (as expected).
"""
)
st.markdown(
    """
    ### Decision Tree with Gender Feature
    """
)
# create basic decision tree without sex feature
decision_tree_ros = create_decision_tree(X_train=X_tros, y_train=y_tros)
decision_tree_smote = create_decision_tree(X_train=X_smote, y_train=y_smote)
# predict targets
y_ros_pred = decision_tree_ros.predict(X_test)
y_smote_pred = decision_tree_smote.predict(X_test)

meth = st.selectbox(
    label="Choose an oversampling method prior to generating decision-trees",
    options=["Random Oversampler", "SMOTE"],
    help="https://imbalanced-learn.org/stable/over_sampling.html"
)
if meth == "Random Oversampler":
    tp = tree.plot_tree(decision_tree_ros,
                        filled=True,
                        rounded=True,
                        class_names=['no_hd','yes_hd'],
                        feature_names=X_encode.columns)
    st.pyplot()
    cm_ros = metrics.confusion_matrix(y_true=y_test, y_pred=y_ros_pred)
    disp_ros = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_ros)
    disp_ros.plot()
    st.pyplot()
else:
    tp = tree.plot_tree(decision_tree_smote,
                        filled=True,
                        rounded=True,
                        class_names=['no_hd','yes_hd'],
                        feature_names=X_encode.columns)
    st.pyplot()
    cm_smote = metrics.confusion_matrix(y_true=y_test, y_pred=y_smote_pred)
    disp_smote = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_smote)
    disp_smote.plot()
    st.pyplot()

f1 = 84.01
auc = 0.81
f1_ros, auc_ros = calculate_metrics(y_test, y_ros_pred)
f1_smote, auc_smote = calculate_metrics(y_test, y_smote_pred)

col1, col2, col3 = st.columns(3, gap="medium")
col1.write("🙅🏻‍♀️ No Oversampling")
with col1:
     st.metric(label="F-Score", value=f"{f1}%")
     st.metric(label="AUC", value=auc)
col2.write("🎲 Random Oversampling")
with col2:
     st.metric(label="F-Score", value=f"{f1_ros}%")
     st.metric(label="AUC", value=auc_ros)
col3.write("🏘 SMOTE")
with col3:
     st.metric(label="F-Score", value=f"{f1_smote}%")
     st.metric(label="AUC", value=auc_smote)

st.markdown(
"""Interestingley, the F-Score and AUC of our model have reduced after applying SMOTE to our training data
when gender information is considered. This might be due to the inbalance in gender distribution
    in both classes but more relevantly in the "no hd" class.
"""
)
st.markdown(
"""
### Gender-specific Decision Trees - Female
"""
)
data_male = data[data['sex'] == 1].copy()
data_female = data[data['sex'] == 0].copy()
X_female = data_female.iloc[:,:13].copy()
y_female = data_female['hd'].copy()
# encode categorial values
X_encode_female = pd.get_dummies(X_female, columns=['cp','restecg','slope','thal'])

# split training and test data
X_train_female, X_test_female, y_train_female, y_test_female = \
model_selection.train_test_split(X_encode_female, y_female, random_state=1, test_size=0.25)

train_data_female = pd.concat([X_train_female, y_train_female], axis=1)
# create basic decision tree
female_class_distr = train_data_female.hd.value_counts()
female_class_distr = female_class_distr.rename(index={0: 'No', 1: 'Yes'})
# plot
# set height
vals = female_class_distr.tolist()
labels = list(female_class_distr.index)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(labels, vals, color = 'darkviolet', width = 0.25)
# look
plt.title('Class Distribution - Females')
plt.xlabel('hd', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks(x, list(female_class_distr.index))

st.pyplot()

st.markdown(
"""
With $$109$$ total counts of diagnosis "no heart disease" and $$36$$ total counts of diagnosis
"heart disease", class labels in our training dataset seem to be shifted towards
a negative diagnosis, which is in contrast to the whole dataset.
This skewness seems more severe than for the dataset where both genders are represented.
Therefore we would expect to seriously improve the performance of our decision-tree classifier by oversampling.
"""
)
Xf_tros, yf_tros = ros.fit_resample(X_train_female, y_train_female)
Xf_smote, yf_smote = SMOTE(random_state=2023).fit_resample(X_train_female, y_train_female)

decision_tree_fros = create_decision_tree(X_train=Xf_tros, y_train=yf_tros)
decision_tree_fsmote = create_decision_tree(X_train=Xf_smote, y_train=yf_smote)
# predict targets
yf_ros_pred = decision_tree_ros.predict(X_test_female)
yf_smote_pred = decision_tree_smote.predict(X_test_female)

methf = st.selectbox(
    label="Choose an oversampling method prior to generating decision-trees (females)",
    options=["Random Oversampler", "SMOTE"],
    help="https://imbalanced-learn.org/stable/over_sampling.html"
)
if methf == "Random Oversampler":
    tp = tree.plot_tree(decision_tree_fros,
                        filled=True,
                        rounded=True,
                        class_names=['no_hd','yes_hd'],
                        feature_names=X_encode_female.columns)
    st.pyplot()
    cm_ros_f = metrics.confusion_matrix(y_true=y_test_female, y_pred=yf_ros_pred)
    disp_ros_f = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_ros_f)
    disp_ros_f.plot()
    st.pyplot()
else:
    tp = tree.plot_tree(decision_tree_fsmote,
                        filled=True,
                        rounded=True,
                        class_names=['no_hd','yes_hd'],
                        feature_names=X_encode_female.columns)
    st.pyplot()
    cm_smote_f = metrics.confusion_matrix(y_true=y_test_female, y_pred=yf_smote_pred)
    disp_smote_f = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_smote_f)
    disp_smote_f.plot()
    st.pyplot()

f1 = 52.63
auc = 0.68
f1_ros_f, auc_ros_f = calculate_metrics(y_test_female, yf_ros_pred)
f1_smote_f, auc_smote_f = calculate_metrics(y_test_female, yf_smote_pred)

col1, col2, col3 = st.columns(3, gap="medium")
col1.write("🙅🏻‍♀️ No Oversampling")
with col1:
     st.metric(label="F-Score", value=f"{f1}%")
     st.metric(label="AUC", value=auc)
col2.write("🎲 Random Oversampling")
with col2:
     st.metric(label="F-Score", value=f"{f1_ros_f}%")
     st.metric(label="AUC", value=auc_ros_f)
col3.write("🏘 SMOTE")
with col3:
     st.metric(label="F-Score", value=f"{f1_smote_f}%")
     st.metric(label="AUC", value=auc_smote_f)

st.markdown(
    """As expected, we see a strong improvement of our model with both oversampling methods.
    """)
st.markdown(
"""
### Gender-specific Decision Trees - Male
"""
)
X_male = data_male.iloc[:,:13].copy()
y_male = data_male['hd'].copy()
# encode categorial values
X_encode_male = pd.get_dummies(X_male, columns=['cp','restecg','slope','thal'])

# split training and test data
X_train_male, X_test_male, y_train_male, y_test_male = \
model_selection.train_test_split(X_encode_male, y_male, random_state=1, test_size=0.25)


train_data_male = pd.concat([X_train_male, y_train_male], axis=1)
# create basic decision tree
male_class_distr = train_data_male.hd.value_counts()
male_class_distr = male_class_distr.rename(index={0: 'No', 1: 'Yes'})
# plot
# set height
vals = male_class_distr.tolist()
labels = list(male_class_distr.index)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(labels, vals, color = 'darkviolet', width = 0.25)
# look
plt.title('Class Distribution - Males')
plt.xlabel('hd', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks(x, list(male_class_distr.index))

st.pyplot()

st.markdown(
"""
With $$337$$ total counts of diagnosis "no heart disease" and $$207$$ total counts of diagnosis
"heart disease", class labels in our training dataset seem to be shifted towards
a positive diagnosis, which can also be seen in the whole dataset.
This skewness also seems more severe than for the dataset where both genders are represented.
Therefore we would expect to improve the performance of our decision-tree classifier by oversampling.
"""
)
Xm_tros, ym_tros = ros.fit_resample(X_train_male, y_train_male)
Xm_smote, ym_smote = SMOTE(random_state=2023).fit_resample(X_train_male, y_train_male)

decision_tree_mros = create_decision_tree(X_train=Xm_tros, y_train=ym_tros)
decision_tree_msmote = create_decision_tree(X_train=Xm_smote, y_train=ym_smote)
# predict targets
ym_ros_pred = decision_tree_ros.predict(X_test_male)
ym_smote_pred = decision_tree_smote.predict(X_test_male)

methm = st.selectbox(
    label="Choose an oversampling method prior to generating decision-trees (males)",
    options=["Random Oversampler", "SMOTE"],
    help="https://imbalanced-learn.org/stable/over_sampling.html"
)
if methm == "Random Oversampler":
    tp = tree.plot_tree(decision_tree_mros,
                        filled=True,
                        rounded=True,
                        class_names=['no_hd','yes_hd'],
                        feature_names=X_encode_male.columns)
    st.pyplot()
    cm_ros_m = metrics.confusion_matrix(y_true=y_test_male, y_pred=ym_ros_pred)
    disp_ros_m = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_ros_m)
    disp_ros_m.plot()
    st.pyplot()
else:
    tp = tree.plot_tree(decision_tree_msmote,
                        filled=True,
                        rounded=True,
                        class_names=['no_hd','yes_hd'],
                        feature_names=X_encode_male.columns)
    st.pyplot()
    cm_smote_m = metrics.confusion_matrix(y_true=y_test_male, y_pred=ym_smote_pred)
    disp_smote_m = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_smote_m)
    disp_smote_m.plot()
    st.pyplot()

f1 = 76.32
auc = 0.7
f1_ros_m, auc_ros_m = calculate_metrics(y_test_male, ym_ros_pred)
f1_smote_m, auc_smote_m = calculate_metrics(y_test_male, ym_smote_pred)

col1, col2, col3 = st.columns(3, gap="medium")
col1.write("🙅🏻‍♀️ No Oversampling")
with col1:
     st.metric(label="F-Score", value=f"{f1}%")
     st.metric(label="AUC", value=auc)
col2.write("🎲 Random Oversampling")
with col2:
     st.metric(label="F-Score", value=f"{f1_ros_m}%")
     st.metric(label="AUC", value=auc_ros_m)
col3.write("🏘 SMOTE")
with col3:
     st.metric(label="F-Score", value=f"{f1_smote_m}%")
     st.metric(label="AUC", value=auc_smote_m)

st.markdown(
    """As expected, we see a strong improvement of our model with both oversampling methods.
        Quite suprisingly, the random oversampling method preformed even better than SMOTE.
    """)

st.markdown(
    """
    ## Conclusion
    - Classifier applied on **data without gender feature** could be slighlty improved by SMOTE,
        however this increase in perfomance is rather small since the data is **not drastically skewed**.
    - The peformance of the classifier applied on **data with gender feature** decreased, which could be 
        explained by the **inbalance of gender distribution** in the whole dataset.
    - In **females**, the class distribution is **skewed towards "no heart disease"**. Performance of the classifier
        could be improved by over $$20\%$$ with both oversampling methods.
    - In **males**, the class distribution is **skewed towards "heart disease"**. Performance of the classifier
        could be improved by both oversampling methods, however random oversampling performed better.
    - We achieved the best model performance for the male-specific decision tree ($$88.43\%$$)
    """)

