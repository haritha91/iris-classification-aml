import argparse
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

from azureml.core import Workspace, Dataset, Experiment, Run

# set max tree size param
parser = argparse.ArgumentParser()
parser.add_argument('--max_tree_depth', type=int, dest='max_tree_depth', default=3, help='maximum tree depth of decision tree')
args = parser.parse_args()
max_tree_depth = args.max_tree_depth

# Get the experiment run context
run = Run.get_context()

# # Load workspace
# ws = Workspace.from_config()
# print(ws.name, "loaded")


# load iris dataset
print("Loading Iris dataset...")
# ds = Dataset.get_by_name(ws,'iris-dataset')
ds = run.input_datasets['iris-dataset']
iris_df = ds.to_pandas_dataframe()

feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
class_names = ['setosa', 'versicolor', 'virginica']
iris_df.head()

# train test split
train, test = train_test_split(iris_df, test_size = 0.25, stratify = iris_df['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

# Training decision tree model
print("Training a decision tree model with a maximum tree depth of ", max_tree_depth)
run.log('Max Tree Depth', int(max_tree_depth))
decisionTree_model = DecisionTreeClassifier(max_depth = int(max_tree_depth), random_state = 1)
decisionTree_model.fit(X_train,y_train)


# plot the decision tree
fig = plt.figure(figsize = (10,8))
plt.title("Decision tree visualization")
plot_tree(decisionTree_model, feature_names = feature_names, class_names = class_names, filled = True);
run.log_image(name='Decision tree', plot = fig)
plt.show()


#calculate accuracy
prediction= decisionTree_model.predict(X_test)
accuracy = metrics.accuracy_score(prediction,y_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(accuracy))
run.log('Accuracy', np.float(accuracy))


#confusion matrix
disp = metrics.plot_confusion_matrix(decisionTree_model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
disp.ax_.set_title('Decision Tree Confusion matrix, without normalization');
run.log_image(name = "confusion matrix", plot = disp)
plt.show()

os.makedirs('outputs', exist_ok=True)
# Save trained model in the outputs directory
joblib.dump(value==decisionTree_model, filename='outputs/iris_model.pkl')

run.complete()