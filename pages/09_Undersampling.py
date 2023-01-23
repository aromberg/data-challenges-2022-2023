import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pacmap

import imblearn.under_sampling as us

st.set_page_config(page_title="Undersampling")

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


data = pd.read_csv("data/data_hr.csv")
########################### Undersamoling specific ####################################################

data = pd.get_dummies(data, columns=['cp','restecg','slope','thal'])
all = data


female = data[data["sex"] == 0]
male = data[data["sex"] == 1] 

nr_males = len(male)
nr_females = len(female)
nr_sick_males = len(male[male["hd"] == 1])
nr_sick_females = len(female[female["hd"] == 1])

st.markdown(
f"# Undersampling\n"
f"This page is an extention of the classification page, in which we investigate wether undersampling can improve the classsification results.\n"
f"The imbalance in the dataset is two fold. Ther is inter class imbalance ie. more males then females and there is intra class imbalance ie. more people with heart desease then without.\n"
f"The first setp in imbalance correction, would be to check the severity of the Problem. In The imputed dataset there are {nr_males} males and {nr_females} females.\n"
f"There are {nr_sick_males} males with heart desease and {nr_males - nr_sick_males} without heart desease.\n"
f"The in the womens group there are {nr_sick_females} people with heart desease and {nr_females + nr_sick_females} without heart desease.\n"
)

st.markdown(
    f"Since females with heartdesease are the smallest group we first balanced the within class imbalance.\n"
    f"In our case it was more important to keep the overall classes structure, then cleaning the data or focusing on the border regions.\n"
    f"Therefore the best undersampling method in this case would be random undersampling, because it has the best chances, of not compromising the classes structure\n"
    f"Prototype generation is not viable in this case, because there is no confirmation, that k-Means can capture the subclasses correctly\n"
    f"For the inter classs undersampling the best method would be the NearMiss 3 method, because our goal is to find differences between the classses\n"
    f"consequently it is very important to keep as much information as possible on the whole border region\n"
)


random_undersampler = us.RandomUnderSampler(random_state=0)
f_resampled, f_hd = random_undersampler.fit_resample(female, female["hd"])
m_resampled, m_hd = random_undersampler.fit_resample(male, male["hd"])
new_data = pd.concat([m_resampled,f_resampled])

near_miss_model = us.NearMiss(version=3)
data, inter_hd = near_miss_model.fit_resample(new_data,new_data["sex"])

female = data[data["sex"] == 0]
male = data[data["sex"] == 1] 

nr_males = len(male)
nr_females = len(female)
nr_sick_males = len(male[male["hd"] == 1])
nr_sick_females = len(female[female["hd"] == 1])

st.markdown(
    f"The new Data set has {nr_males} males, of which {nr_sick_males} have heartdesease and {nr_females} of which {nr_sick_females} have heartdesease.\n"
)

pacmaped = pacmap.PaCMAP(n_components=2, n_neighbors=None).fit_transform(all.to_numpy())
pacmaped_transformed = pd.DataFrame(pacmaped)

pacmaped_transformed["sex"] = all["sex"]
figure2 = plt.figure()
if (len(pd.unique(all["sex"])) > 1):
    plt.scatter((pacmaped_transformed[pacmaped_transformed.sex == 0])[0], (pacmaped_transformed[pacmaped_transformed.sex == 0])[1])
    plt.scatter((pacmaped_transformed[pacmaped_transformed.sex == 1])[0], (pacmaped_transformed[pacmaped_transformed.sex == 1])[1])
else:
    plt.scatter((pacmaped_transformed)[0], (pacmaped_transformed)[1])
st.pyplot(figure2)


pacmaped = pacmap.PaCMAP(n_components=2, n_neighbors=None).fit_transform(data.to_numpy())
pacmaped_transformed = pd.DataFrame(pacmaped)

pacmaped_transformed["sex"] = all["sex"]
figure2 = plt.figure()
if (len(pd.unique(all["sex"])) > 1):
    plt.scatter((pacmaped_transformed[pacmaped_transformed.sex == 0])[0], (pacmaped_transformed[pacmaped_transformed.sex == 0])[1])
    plt.scatter((pacmaped_transformed[pacmaped_transformed.sex == 1])[0], (pacmaped_transformed[pacmaped_transformed.sex == 1])[1])
else:
    plt.scatter((pacmaped_transformed)[0], (pacmaped_transformed)[1])
st.pyplot(figure2)

#######################################################################################################################################

st.markdown(
"""
## Classification with the new Data
First, we again create a decision tree using scikit-learn without considering gender \
as a criterion for decision making.
"""
)

# split features and targets
X_encode = data.iloc[:,data.columns != "hd"].copy()
y = data['hd'].copy()

# encode categorial values
# X_encode = pd.get_dummies(X, columns=['cp','restecg','slope','thal'])

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

counts_hd_no = len(data[data['hd']==0])
counts_hd_yes = len(data[data['hd']==1])

accuracy_without_sex = metrics.accuracy_score(y_test, y_pred_without_sex)
accuracy_without_sex_str = str(100 * round(accuracy_without_sex, 4))
recall_without_sex = metrics.recall_score(y_test, y_pred_without_sex)
recall_without_sex_str = str(round(100 * recall_without_sex, 2))

st.markdown(
    f"Accuracy: {accuracy_without_sex_str}, Recall: {recall_without_sex_str}"
)

################################################################################

st.markdown(
"""
## Descisiontree with gender feature
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

accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_str = str(100 * round(accuracy, 4))
recall = metrics.recall_score(y_test, y_pred)
recall_str = str(round(100 * recall, 2))

st.markdown(
"""
The accuray is {}% and the recall is {}%.
""".format(accuracy_str, recall_str)
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
"""
)
# female

# split features and targets
X_female = data_female.iloc[:,data_female.columns != "hd"].copy()
y_female = data_female['hd'].copy()

# encode categorial values
# X_encode_female = pd.get_dummies(X_female, columns=['cp','restecg','slope','thal'])

X_encode_female = X_female

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

# calculate accuracy
accuracy_female = metrics.accuracy_score(y_test_female, y_pred_female)
accuracy_female_str = str(100 * round(accuracy_female, 4))
recall_female = metrics.recall_score(y_test_female, y_pred_female)
recall_female_str = str(round(100 * recall_female, 2))

st.markdown(
"""
The accuracy is {}% and the recall is {}%.
""".format(accuracy_female_str, recall_female_str)
)

# male

# split features and targets
X_male = data_male.iloc[:,data_male.columns != "hd"].copy()
y_male = data_male['hd'].copy()

# encode categorial values
# X_encode_male = pd.get_dummies(X_male, columns=['cp','restecg','slope','thal'])

X_encode_male = X_male

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

# calculate accuracy
accuracy_male = metrics.accuracy_score(y_test_male, y_pred_male)
accuracy_male_str = str(100 * round(accuracy_male, 4))
recall_male = metrics.recall_score(y_test_male, y_pred_male)
recall_male_str = str(round(100 * recall_male, 2))

st.markdown(
"""
The accuracy is {}% and the recall is {}%.
""".format(accuracy_male_str, recall_male_str)
)

st.markdown(
"""
## Conclusion
With the undesampled data the gender specific trees show better performance, while the descisiontree considering sex does not improve.
"""
)