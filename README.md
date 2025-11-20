# VKPyKit

This package contains several functions used for Exploratory Data Analysis, Linear Regression, Decision Trees. Instead of writing those everytime, just wanted to put them in a package and reuse myself. 

Never liked to write same code over and over, PyPI allows to reuse your code. 

~~~
## Using my own Pythn libraries for Exploratory Data Analysis and Decision Tree Modeling.
## 1. EDA - Exploratory Data Analysis
## 2. DT - Decision Tree
## Easier to write once and use as packaged functions. 

from VKPyKit.EDA import *
from VKPyKit.DT import *
from VKPyKit.LR import *

EDA= EDA()
DT = DT()
LR = LR()

# Example Model Performance Classification
DT.model_performance_classification(
            myDecisionTreeClassifierModel,
            myPredictors,
            myExpected,
            printall=True,
            title='My Decision Tree Model')

# Example Plot Confusion Matrix
DT.plot_confusion_matrix(
            myDecisionTreeClassifierModel,
            myPredictors,
            myExpected,
            title='My Decision Tree Model')

# Example Decision Tree Tuning
DT.tune_decision_tree(
            X_train=myX_train,
            y_train=myY_train,
            X_test=myX_test,
            y_test=myY_test,
            max_depth_v=(2, 11, 2),
            max_leaf_nodes_v=(10, 51, 10),
            min_samples_split_v=(10, 51, 10),
            printall=True,
            sortresultby=['F1Difference'],
            sortbyAscending=False)

# Example Histogram Boxplot All
EDA.histogram_boxplot_all(data=myData,
                              figsize=(15, 10),
                              bins=10,
                              kde=True)

LR.linear_regression_model(data=myData,
                              predictors=myPredictors,
                              target=myTarget,
                              printall=True,
                              title='My Linear Regression Model')

~~~
Use it as is, if you find issues or have more such functions, please contribute on github. 


[Source on Github](https://github.com/assignarc/VKPyKit)