#
# %%
# to load and manipulate data
import pandas as pd
import numpy as np

# to visualize data
import matplotlib.pyplot as plt
import seaborn as sns

# to split data into training and test sets
from sklearn.model_selection import train_test_split

# to build decision tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# to tune different models
from sklearn.model_selection import GridSearchCV

# to compute classification metrics
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

from tests.VK import *




# %%
# loading data into a pandas dataframe
credit_approval = pd.read_csv("credit_card_approval.csv")

# %%
# creating a copy of the data
data = credit_approval.copy()

# %% [markdown]
# # **Data Overview**

# %% [markdown]
# ### Viewing the first and last 5 rows of the dataset

# %%
data.head(5)

# %%
data.tail(5)

# %% [markdown]
# ### Checking the shape of the dataset.

# %%
data.shape

# %% [markdown]
# - The dataset has 2500 rows and 7 columns.

# %% [markdown]
# ### Checking the attribute types

# %%
data.info()

# %% [markdown]
# - There are 3 numerical and 4 categorical variables in the data.
# - Employed and PrioriDefault, although interpreted here as numerical, are categorical variables that are encoded by default.

# %% [markdown]
# ### Checking the statistical summary

# %%
data.describe(include="all")

# %% [markdown]
# - ~60% of the applications were approved.
# - The average age of applicants is ~37 years.
# - On average, applicants have a credit score of ~703.
# - Applicants earn ~75k dollars annually, with ~25% earning more than $100k annually.
# - At least 25% of the applicants have a prior default history.

# %% [markdown]
# ### Checking for missing values

# %%
# checking for null values
data.isnull().sum()

# %% [markdown]
# - There are no missing values in the dataset.

# %% [markdown]
# ### Checking for duplicate values

# %%
# checking for duplicate values
data.duplicated().sum()

# %% [markdown]
# * There are no duplicate values in the data.

# %% [markdown]
# # **Exploratory Data Analysis**

# %% [markdown]
# ### Univariate Analysis

# %%
# defining the figure size
plt.figure(figsize=(15, 10))

# defining the list of numerical features to plot
num_features = ['Age', 'Annual Income', 'Credit Score']

# plotting the histogram for each numerical feature
for i, feature in enumerate(num_features):
    plt.subplot(3, 3, i+1)    # assign a subplot in the main plot
    sns.histplot(data=data, x=feature)    # plot the histogram

plt.tight_layout();   # to add spacing between plots

# %% [markdown]
# - **Age** and **Annual Income** exhibit a right-skewed distribution.
# - **Credit Score** exhibits a left-skewed distribution.

# %%
# defining the figure size
plt.figure(figsize=(15, 10))

# plotting the boxplot for each numerical feature
for i, feature in enumerate(num_features):
    plt.subplot(3, 3, i+1)    # assign a subplot in the main plot
    sns.boxplot(data=data, x=feature)    # plot the histogram

plt.tight_layout();    # to add spacing between plots

# %% [markdown]
# * There are outliers in all the numerical attributes in the data.

# %%
# checking the distribution of the categories in Approval
print(100*data['Approval'].value_counts(normalize=True), '\n')

# plotting the count plot for Approval
sns.countplot(data=data, x='Approval');

# %% [markdown]
# - ~60% of the credit card applications in the data have been approved.

# %%
# checking the distribution of the categories in PriorDefault
print(100*data['PriorDefault'].value_counts(normalize=True), '\n')

# plotting the count plot for PriorDefault
sns.countplot(data=data, x='Approval');

# %% [markdown]
# - 70% of the applicants have no prior defaults.
# 
# 
# 
# 
# 
# 
# 
# 

# %%
# checking the distribution of the categories in Employed
print(100*data['Employed'].value_counts(normalize=True), '\n')

# plotting the count plot for Employed
sns.countplot(data=data, x='Employed');

# %% [markdown]
# - ~70% of applicants are currently employed.

# %%
# checking the distribution of the categories in Gender
print(100*data['Gender'].value_counts(normalize=True), '\n')

# plotting the count plot for Gender
sns.countplot(data=data, x='Gender');

# %% [markdown]
# - Almost half the applicants are women.

# %% [markdown]
# ### Bivariate Analysis

# %%
# defining the size of the plot
plt.figure(figsize=(12, 7))

# plotting the heatmap for correlation
sns.heatmap(
    data[num_features].corr(),annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
);

# %% [markdown]
# - As the age of the applicants increases, their annual incomes increase.
# - The correlation between annual income and credit score is positive but not as strong as that between annual income and age.

# %%
# Scatter plot matrix
plt.figure(figsize=(12, 8))
sns.pairplot(data, vars=num_features, hue='Approval', diag_kind='kde');

# %% [markdown]
# - Applicants with a credit score of more than 640 are more likely to have their application approved.

# %%
# Credit Score vs Approved (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Approval', y='Credit Score');

# %% [markdown]
# - In general, applicants with higher credit scores have higher chances of getting credit card approval.
#     - For a credit score of more than 600, you are likely to get a credit card.
# - The range of credit scores of applicants whose applications were rejected is wider.

# %%
# Income vs Approved (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Approval', y='Annual Income')
plt.title('Income vs Approved (Boxplot)')
plt.show()

# %% [markdown]
# - Applicants with high annual incomes have slightly higher chances of getting credit cards approved.

# %%
# creating a crosstab for Approval vs PriorDefault
tab = pd.crosstab(
    data['PriorDefault'],
    data['Approval'],
    normalize='index'    # normalizing by dividing each row by its row total
).sort_values(by='No', ascending=False)    # sorting the resulting crosstab


# Plot the stacked bar chart
tab.plot(kind='bar', stacked=True, figsize=(7, 5))    # creating a stacked bar chart from the normalized crosstab
plt.xlabel('PriorDefault')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Approval');    # adding a legend for the 'Approval' column

# %% [markdown]
# - Applicants with a prior default history have a higher chance of having their application rejected.

# %%
# creating a crosstab for Approval vs Employed
tab = pd.crosstab(
    data['Employed'],
    data['Approval'],
    normalize='index'    # normalizing by dividing each row by its row total
).sort_values(by='No', ascending=False)    # sorting the resulting crosstab


# Plot the stacked bar chart
tab.plot(kind='bar', stacked=True, figsize=(7, 5))    # creating a stacked bar chart from the normalized crosstab
plt.xlabel('Employed')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Approval');    # adding a legend for the 'Approval' column

# %% [markdown]
# - ~70% of applicants who are currently not employed have their credit card applications rejected, while ~70% of applicants who are currently employed have their credit card applications approved.

# %% [markdown]
# # **Data Preparation for Modeling**

# %%
# defining the explanatory (independent) and response (dependent) variables
X = data.drop(["Approval"], axis=1)
y = data["Approval"]

# %%
# creating dummy variables
X = pd.get_dummies(X, columns=X.select_dtypes(include=["object", "category"]).columns.tolist(), drop_first=True)

# specifying the datatype of the independent variables data frame
X = X.astype(float)

X.head()

# %%
# label encoding the response variable
y = y.map({'Yes': 1, 'No': 0})

y.head()

# %%
# splitting the data in an 80:20 ratio for train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)    # stratify ensures that the training and test sets have a similar distribution of the response variable

# %%
print("Shape of training set:", X_train.shape)
print("Shape of test set:", X_test.shape, '\n')
print("Percentage of classes in training set:")
print(100*y_train.value_counts(normalize=True), '\n')
print("Percentage of classes in test set:")
print(100*y_test.value_counts(normalize=True))

# %% [markdown]
# # **Model Building**

# %% [markdown]
# ### Decision Tree (sklearn default)
# 

# %%
# creating an instance of the decision tree model
dtree1 = DecisionTreeClassifier(random_state=42)    # random_state sets a seed value and enables reproducibility

# fitting the model to the training data
dtree1.fit(X_train, y_train)

# %% [markdown]
# #### Model Evaluation

# %% [markdown]
# We define a utility function to collate all the metrics into a single data frame, and another to plot the confusion matrix.

# %%
# defining a function to compute different metrics to check performance of a classification model built using sklearn
# def model_performance_classification(model, predictors, target):
#     """
#     Function to compute different metrics to check classification model performance

#     model: classifier
#     predictors: independent variables
#     target: dependent variable
#     """

#     # predicting using the independent variables
#     pred = model.predict(predictors)

#     acc = accuracy_score(target, pred)  # to compute Accuracy
#     recall = recall_score(target, pred)  # to compute Recall
#     precision = precision_score(target, pred)  # to compute Precision
#     f1 = f1_score(target, pred)  # to compute F1-score

#     # creating a dataframe of metrics
#     df_perf = pd.DataFrame(
#         {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
#         index=[0],
#     )

#     return df_perf

# %%
# def plot_confusion_matrix(model, predictors, target):
#     """
#     To plot the confusion_matrix with percentages

#     model: classifier
#     predictors: independent variables
#     target: dependent variable
#     """
#     # Predict the target values using the provided model and predictors
#     y_pred = model.predict(predictors)

#     # Compute the confusion matrix comparing the true target values with the predicted values
#     cm = confusion_matrix(target, y_pred)

#     # Create labels for each cell in the confusion matrix with both count and percentage
#     labels = np.asarray(
#         [
#             ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
#             for item in cm.flatten()
#         ]
#     ).reshape(2, 2)    # reshaping to a matrix

#     # Set the figure size for the plot
#     plt.figure(figsize=(6, 4))

#     # Plot the confusion matrix as a heatmap with the labels
#     sns.heatmap(cm, annot=labels, fmt="")

#     # Add a label to the y-axis
#     plt.ylabel("True label")

#     # Add a label to the x-axis
#     plt.xlabel("Predicted label")

# %% [markdown]
# **Note**: We want to maximize the F1 Score to ensure that we reduce both the chances of approving non-credible applications as well as rejecting credible ones.

# %%
ai = VAI.AIFunctions()
ai.plot_confusion_matrix(model=dtree1, predictors=X_train, target=y_train)

# %%


# %%
dtree1_train_perf = vkai.model_performance_classification(dtree1, X_train, y_train)

dtree1_train_perf

# %%
vkai.plot_confusion_matrix(dtree1, X_test, y_test)

# %%
dtree1_test_perf = vkai.model_performance_classification(
    dtree1, X_test, y_test
)
dtree1_test_perf

# %% [markdown]
# - There is a huge difference between the training and test F1 Scores.
# - This indicates that the model is overfitting.

# %% [markdown]
# #### Visualizing the Decision Tree

# %%
# list of feature names in X_train
feature_names = list(X_train.columns)

# set the figure size for the plot
plt.figure(figsize=(20, 20))

# plotting the decision tree
out = tree.plot_tree(
    dtree1,                         # decision tree classifier model
    feature_names=feature_names,    # list of feature names (columns) in the dataset
    filled=True,                    # fill the nodes with colors based on class
    fontsize=9,                     # font size for the node text
    node_ids=False,                 # do not show the ID of each node
    class_names=None,               # whether or not to display class names
)

# add arrows to the decision tree splits if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")    # set arrow color to black
        arrow.set_linewidth(1)          # set arrow linewidth to 1

# displaying the plot
plt.show()

# %% [markdown]
# - We can observe that this is a very complex tree.

# %%
# printing a text report showing the rules of a decision tree
print(
    tree.export_text(
        dtree1,    # specify the model
        feature_names=feature_names,    # specify the feature names
        show_weights=True    # specify whether or not to show the weights associated with the model
    )
)

# %% [markdown]
# ### Decision Tree (Pre-pruning)

# %%
best_estimator = vkai.tune_decision_tree(X_train, y_train, X_test, y_test,
                                 max_depth_v=(2, 11, 2),
                                 max_leaf_nodes_v=(10, 51, 10),
                                 min_samples_split_v=(10, 51, 10),
                                 printall=False,
                                 sortresultby=['score_diff'])

# %%
# creating an instance of the best model
dtree2 = best_estimator

# fitting the best model to the training data
dtree2.fit(X_train, y_train)

# %% [markdown]
# #### Model Evaluation

# %%
vkdt.plot_confusion_matrix(dtree2, X_train, y_train)

# %%
dtree2_train_perf = vkdt.model_performance_classification(
    dtree2, X_train, y_train
)
dtree2_train_perf

# %%
vkdt.plot_confusion_matrix(dtree2, X_test, y_test)

# %%
dtree2_test_perf = vkdt.model_performance_classification(
    dtree2, X_test, y_test
)
dtree2_test_perf

# %% [markdown]
# - The training and test scores are very close to each other, indicating a generalized performance.

# %% [markdown]
# #### Visualizing the Decision Tree

# %%
# list of feature names in X_train
feature_names = list(X_train.columns)

# set the figure size for the plot
plt.figure(figsize=(20, 20))

# plotting the decision tree
out = tree.plot_tree(
    dtree2,                         # decision tree classifier model
    feature_names=feature_names,    # list of feature names (columns) in the dataset
    filled=True,                    # fill the nodes with colors based on class
    fontsize=9,                     # font size for the node text
    node_ids=False,                 # do not show the ID of each node
    class_names=None,               # whether or not to display class names
)

# add arrows to the decision tree splits if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")    # set arrow color to black
        arrow.set_linewidth(1)          # set arrow linewidth to 1

# displaying the plot
plt.show()

# %% [markdown]
# - This is a far less complex tree than the previous one.
# - We can observe the decision rules much more clearly in the plot.

# %%
# printing a text report showing the rules of a decision tree
print(
    tree.export_text(
        dtree2,    # specify the model
        feature_names=feature_names,    # specify the feature names
        show_weights=True    # specify whether or not to show the weights associated with the model
    )
)

# %% [markdown]
# ### Decision Tree (Post-pruning)

# %%
# Create an instance of the decision tree model
clf = DecisionTreeClassifier(random_state=42)

# Compute the cost complexity pruning path for the model using the training data
path = clf.cost_complexity_pruning_path(X_train, y_train)

# Extract the array of effective alphas from the pruning path
ccp_alphas = abs(path.ccp_alphas)

# Extract the array of total impurities at each alpha along the pruning path
impurities = path.impurities

# %%
pd.DataFrame(path)

# %%
# Create a figure
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the total impurities versus effective alphas, excluding the last value,
# using markers at each data point and connecting them with steps
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")

# Set the x-axis label
ax.set_xlabel("Effective Alpha")

# Set the y-axis label
ax.set_ylabel("Total impurity of leaves")

# Set the title of the plot
ax.set_title("Total Impurity vs Effective Alpha for training set");

# %% [markdown]
# - Next, we train a decision tree using the effective alphas.
# 
# - The last value in `ccp_alphas` is the alpha value that prunes the whole tree,
# leaving the corresponding tree with one node.

# %%
# Initialize an empty list to store the decision tree classifiers
clfs = []

# Iterate over each ccp_alpha value extracted from cost complexity pruning path
for ccp_alpha in ccp_alphas:
    # Create an instance of the DecisionTreeClassifier
    clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Append the trained classifier to the list
    clfs.append(clf)

# Print the number of nodes in the last tree along with its ccp_alpha value
print(
    "Number of nodes in the last tree is {} with ccp_alpha {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

# %% [markdown]
# - Moving ahead, we remove the last element in
# ``clfs`` and ``ccp_alphas`` as it corresponds to a trivial tree with only one
# node.

# %%
# Remove the last classifier and corresponding ccp_alpha value from the lists
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# Extract the number of nodes in each tree classifier
node_counts = [clf.tree_.node_count for clf in clfs]

# Extract the maximum depth of each tree classifier
depth = [clf.tree_.max_depth for clf in clfs]

# Create a figure and a set of subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 7))

# Plot the number of nodes versus ccp_alphas on the first subplot
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("Alpha")
ax[0].set_ylabel("Number of nodes")
ax[0].set_title("Number of nodes vs Alpha")

# Plot the depth of tree versus ccp_alphas on the second subplot
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("Alpha")
ax[1].set_ylabel("Depth of tree")
ax[1].set_title("Depth vs Alpha")

# Adjust the layout of the subplots to avoid overlap
fig.tight_layout()

# %%
train_f1_scores = []  # Initialize an empty list to store F1 scores for training set for each decision tree classifier

# Iterate through each decision tree classifier in 'clfs'
for clf in clfs:
    # Predict labels for the training set using the current decision tree classifier
    pred_train = clf.predict(X_train)

    # Calculate the F1 score for the training set predictions compared to true labels
    f1_train = f1_score(y_train, pred_train)

    # Append the calculated F1 score to the train_f1_scores list
    train_f1_scores.append(f1_train)

# %%
test_f1_scores = []  # Initialize an empty list to store F1 scores for test set for each decision tree classifier

# Iterate through each decision tree classifier in 'clfs'
for clf in clfs:
    # Predict labels for the test set using the current decision tree classifier
    pred_test = clf.predict(X_test)

    # Calculate the F1 score for the test set predictions compared to true labels
    f1_test = f1_score(y_test, pred_test)

    # Append the calculated F1 score to the test_f1_scores list
    test_f1_scores.append(f1_test)


# %%
# Create a figure
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlabel("Alpha")  # Set the label for the x-axis
ax.set_ylabel("F1 Score")  # Set the label for the y-axis
ax.set_title("F1 Score vs Alpha for training and test sets")  # Set the title of the plot

# Plot the training F1 scores against alpha, using circles as markers and steps-post style
ax.plot(ccp_alphas, train_f1_scores, marker="o", label="training", drawstyle="steps-post")

# Plot the testing F1 scores against alpha, using circles as markers and steps-post style
ax.plot(ccp_alphas, test_f1_scores, marker="o", label="test", drawstyle="steps-post")

ax.legend();  # Add a legend to the plot

# %%
# creating the model where we get highest test F1 Score
index_best_model = np.argmax(test_f1_scores)

# selcting the decision tree model corresponding to the highest test score
dtree3 = clfs[index_best_model]
print(dtree3)

# %% [markdown]
# #### Model Evaluation

# %%
vkdt.plot_confusion_matrix(dtree3, X_train, y_train)

# %%
dtree3_train_perf = vkdt.model_performance_classification(
    dtree3, X_train, y_train
)
dtree3_train_perf

# %%
vkdt.plot_confusion_matrix(dtree3, X_test, y_test)

# %%
dtree3_test_perf = vkdt.model_performance_classification(
    dtree3, X_test, y_test
)
dtree3_test_perf

# %% [markdown]
# - The test score is greater than the training score, indicating a generalized performance.

# %% [markdown]
# #### Visualizing Decision Tree

# %%
# list of feature names in X_train
feature_names = list(X_train.columns)

# set the figure size for the plot
plt.figure(figsize=(10, 7))

# plotting the decision tree
out = tree.plot_tree(
    dtree3,                         # decision tree classifier model
    feature_names=feature_names,    # list of feature names (columns) in the dataset
    filled=True,                    # fill the nodes with colors based on class
    fontsize=9,                     # font size for the node text
    node_ids=False,                 # do not show the ID of each node
    class_names=None,               # whether or not to display class names
)

# add arrows to the decision tree splits if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")    # set arrow color to black
        arrow.set_linewidth(1)          # set arrow linewidth to 1

# displaying the plot
plt.show()

# %% [markdown]
# - This is a far less complex tree than the previous one.
# - The model used only current employment status and prior default history to determine whether the application will be approved or not.

# %%
# printing a text report showing the rules of a decision tree
print(
    tree.export_text(
        dtree3,    # specify the model
        feature_names=feature_names,    # specify the feature names
        show_weights=True    # specify whether or not to show the weights associated with the model
    )
)

# %% [markdown]
# # **Model Performance Comparison and Final Model Selection**

# %%
# training performance comparison

models_train_comp_df = pd.concat(
    [
        dtree1_train_perf.T,
        dtree2_train_perf.T,
        dtree3_train_perf.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree (sklearn default)",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Training performance comparison:")
models_train_comp_df

# %%
# testing performance comparison

models_test_comp_df = pd.concat(
    [
        dtree1_test_perf.T,
        dtree2_test_perf.T,
        dtree3_test_perf.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree (sklearn default)",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Test set performance comparison:")
models_test_comp_df.T

# %% [markdown]
# - Both the pre-pruned and post-pruned decision trees exhibit generalized performances.
# 
# - The post-pruned decision tree has an approx. 3.5% better performance on the test set than the training set.
#     - This model uses only two features for decision-making.
#     - This will result in a low prediction time but it might not be able to perform well on edge cases in unseen data.
# 
# - The pre-pruned decision tree has almost the same performance on training and test sets.
#     - This model uses a few more features for decision-making than the post-pruned decision tree.
#     - This will result in a slightly longer prediction time but it is likely to yield better results on unseen data.
# 
# - We'll move ahead with the pre-pruned decision tree as our final model.

# %% [markdown]
# ### Feature Importance

# %%
# importance of features in the tree building
importances = dtree2.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %% [markdown]
# - Current employment status and prior default history are the most influential attributes in determining credit worthiness.
# - Credit score and annual income are the next best attributes to consider.

# %% [markdown]
# ### Predicting on a single data point

# %%

# choosing a data point
applicant_details = X_test.iloc[:1, :]

# making a prediction
approval_prediction = dtree2.predict(applicant_details)

print(approval_prediction)

# %% [markdown]
# - The model was able to predict in under half a second.
# - Instead of predicting a class (approve/reject), the model can also predict the likelihood of approval.

# %%
# making a prediction
approval_likelihood = dtree2.predict_proba(applicant_details)

print(approval_likelihood[0, 1])

# %% [markdown]
# - This indicates that the model is ~91% confident that the applicant is creditworthy and the application should be approved.

# %% [markdown]
# # **Business Recommendations**

# %% [markdown]
# -  The bank can deploy this model for the initial screening of credit card applications.
# 
# - Instead of outputting an approve or reject, the model can be made to output the likelihood of approval.
# 
# - In case the likelihood of approval is below a certain threshold, say 60%, then the application can be sent for manual inspection.
# 
# - This would reduce the overall TAT for the initial screening.

# %% [markdown]
# <font size=6 color='blue'>Power Ahead</font>
# ___


