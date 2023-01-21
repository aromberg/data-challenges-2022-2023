import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def calculate_metrics(y_test, y_pred):
    """Calculate f1 and auc and return them as rounded strings."""

    f1 = metrics.f1_score(y_test, y_pred)
    f1_str = str(round(100 * f1, 2))

    auc = metrics.roc_auc_score(y_test, y_pred)
    auc_str = str(round(auc, 2))

    return f1_str, auc_str

st.set_page_config(page_title="Classification")

st.markdown(
"""
# Classification for Detecting Heart Disease
## Preprocessing Data
In order not to confuse numerical and categorical features, we made the dataset
a bit more human-readable. Note that we still use the imputed dataset.
""")
data = pd.read_csv("data/data_hr.csv")
st.dataframe(data.head())

def create_decision_tree(data, X_train, y_train, X_test, y_test):
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
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_encode, y, random_state=1, test_size=0.25)

# delete sex feature
data_without_sex = data.drop(['sex'], axis=1)
X_train_without_sex = X_train.drop(['sex'], axis=1)
X_test_without_sex = X_test.drop(['sex'], axis=1)

# create basic decision tree without sex feature
decision_tree_without_sex = create_decision_tree(data_without_sex, X_train_without_sex, y_train, X_test_without_sex, y_test)

st.set_option('deprecation.showPyplotGlobalUse', False)
tp = tree.plot_tree(decision_tree_without_sex,
              filled=True,
              rounded=True,
              class_names=['no_hd','yes_hd'],
              feature_names=X_encode.columns)
st.pyplot()

# predict targets
y_pred_without_sex = decision_tree_without_sex.predict(X_test_without_sex)

st.subheader("Confusion Matrix for Decision Tree without Gender Feature")
cm_without_sex = metrics.confusion_matrix(y_test, y_pred_without_sex)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_without_sex)
disp.plot()
st.pyplot()

st.markdown(
"We evaluated the performance of the decision tree with a confusion matrix."
)

counts_hd_no = len(data[data['hd']==0])
counts_hd_yes = len(data[data['hd']==1])

f1, auc = calculate_metrics(y_test, y_pred_without_sex)

st.markdown(
"We check the balance of the dataset with respect to heart disease. \
The number of records without heard disease is " + str(counts_hd_no) +
" and the number of records with heard disease is " + str(counts_hd_yes) + ". \
The F1-Score is {}% and the AUC is {}".format(f1, auc)
)

################################################################################

st.markdown(
"""
## Decision Tree with Gender Feature
Now we take gender into account to see if it improves the classifier.
""")

# create basic decision tree
decision_tree = create_decision_tree(data, X_train, y_train, X_test, y_test)

# visualize decision tree
tp = tree.plot_tree(decision_tree,
              filled=True,
              rounded=True,
              class_names=['no_hd','yes_hd'],
              feature_names=X_encode.columns)
st.pyplot()

st.markdown(
"If we look at the decision criteria in the tree, we see that gender occurs once."
)

# predict targets
y_pred = decision_tree.predict(X_test)

# calculate confusion matrix
st.subheader("Confusion Matrix for Decision Tree with Gender Feature")
cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
st.pyplot()


f1, auc = calculate_metrics(y_test, y_pred)

st.markdown(
"""
The F1-Score is {}% and the AUC is {}. This means that the performance has \
remained more or less the same. Gender does not seem to be an important \
medical criterion for predicting the disease.
""".format(f1, auc)
)

################################################################################

# count male and female records
counts_male = len(data[data['sex']==1])
counts_female = len(data[data['sex']==0])

# split dataset by gender
data_male = data[data['sex'] == 1].copy()
data_female = data[data['sex'] == 0].copy()

counts_hd_yes_male = len(data_male[data_male['hd']==1])
counts_hd_yes_female = len(data_female[data_female['hd']==1])

st.markdown(
"""
## Gender-specific Decision Trees
Now we create decision trees for both genders separately to see if we can \
improve performance. There are {} females and {} males in the dataset. \
{} females and {} males have heart diseases. The imbalances could strongly influence the \
classifier performances.
""".format(counts_female, counts_male, counts_hd_yes_female, counts_hd_yes_male))

# female

# split features and targets
X_female = data_female.iloc[:,:13].copy()
y_female = data_female['hd'].copy()

# encode categorial values
X_encode_female = pd.get_dummies(X_female, columns=['cp','restecg','slope','thal'])

# split training and test data
X_train_female, X_test_female, y_train_female, y_test_female = \
model_selection.train_test_split(X_encode_female, y_female, random_state=1, test_size=0.25)

# create basic decision tree
decision_tree_female = create_decision_tree(data, X_train_female, y_train_female, X_test_female, y_test_female)

# visualize decision tree
st.subheader('Female Decision Tree')
tp = tree.plot_tree(decision_tree_female,
              filled=True,
              rounded=True,
              class_names=['no_hd','yes_hd'],
              feature_names=X_encode.columns)
st.pyplot()

# predict targets
y_pred_female = decision_tree_female.predict(X_test_female)

st.subheader("Confusion Matrix for Female Decision Tree")
cm_female = metrics.confusion_matrix(y_test_female, y_pred_female)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_female)
disp.plot()
st.pyplot()

f1, auc = calculate_metrics(y_test_female, y_pred_female)

st.markdown(
"""
The F1-Score is {}% and the AUC is {}. The performance has decreased \
significantly.
""".format(f1, auc)
)

# male

# split features and targets
X_male = data_male.iloc[:,:13].copy()
y_male = data_male['hd'].copy()

# encode categorial values
X_encode_male = pd.get_dummies(X_male, columns=['cp','restecg','slope','thal'])

# split training and test data
X_train_male, X_test_male, y_train_male, y_test_male = \
model_selection.train_test_split(X_encode_male, y_male, random_state=1, test_size=0.25)

# create basic decision tree
decision_tree_male = create_decision_tree(data, X_train_male, y_train_male, X_test_male, y_test_male)

# visualize decision tree
st.subheader('Male Decision Tree')
tp = tree.plot_tree(decision_tree_male,
              filled=True,
              rounded=True,
              class_names=['no_hd','yes_hd'],
              feature_names=X_encode.columns)
st.pyplot()

# predict targets
y_pred_male = decision_tree_male.predict(X_test_male)

st.subheader("Confusion Matrix for Male Decision Tree")
cm_male = metrics.confusion_matrix(y_test_male, y_pred_male)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_male)
disp.plot()
st.pyplot()

f1, auc = calculate_metrics(y_test_male, y_pred_male)

st.markdown(
"""
The F1-Score is {}% and the AUC is {}%. A male-specific decision tree also \
failed to improve performance in comparison to gender-unspecific decision \
trees, although less significantly.
""".format(f1, auc)
)

st.markdown(
"""
## Conclusion
Overall, we found that the performance of the classifier could not be improved by \
considering gender in decision making. Gender-specific classifiers reduce performance. \
Possibly the overfit is the reason for this.
"""
)
