import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from azureml.core import Workspace, Dataset, Experiment, Run

ws = Workspace.from_config()
print(ws.name, "loaded")

ds = Dataset.get_by_name(ws,'iris-dataset')
iris_df = ds.to_pandas_dataframe()

feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
class_names = ['setosa', 'versicolor', 'virginica']
iris_df

#train test split
train, test = train_test_split(iris_df, test_size = 0.25, stratify = iris_df['species'], random_state = 42)

X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

decisionTree_model = DecisionTreeClassifier(max_depth = 3, random_state = 1)
decisionTree_model.fit(X_train,y_train)

#calculate accuracy
prediction= decisionTree_model.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

#calculate AUC
y_scores = decisionTree_model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
# run.log('AUC', np.float(auc))


#plot the decision tree
plt.figure(figsize = (10,8))
plot_tree(decisionTree_model, feature_names = feature_names, class_names = class_names, filled = True);
plt.show()

#confusion matrix
disp = metrics.plot_confusion_matrix(decisionTree_model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
disp.ax_.set_title('Decision Tree Confusion matrix, without normalization');
plt.show()

