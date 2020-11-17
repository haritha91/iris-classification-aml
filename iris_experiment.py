import azureml.core
import os
from azureml.core import Experiment, Workspace, Model
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.hyperdrive import GridParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice
from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn


ws = Workspace.from_config()
print("Ready to use Azure ML", azureml.core.VERSION)
print('Ready to work with', ws.name)

# Create an experiment
experiment_name = 'iris_training'
experiment = Experiment(workspace = ws, name = experiment_name)

# Create a folder for the experiment files
experiment_folder = './' + experiment_name
os.makedirs(experiment_folder, exist_ok=True)

print("Experiment:", experiment.name)

#Fetch CPU cluster for computations
cpu_cluster = ComputeTarget(workspace=ws, name='cpu-compute')

# Sample a range of parameter values
params = GridParameterSampling(
    {
        # There's only one parameter, so grid sampling will try each value - with multiple parameters it would try every combination
        '--max_tree_depth': choice(2, 3, 4, 5, 6)
    }
)

# Set evaluation policy to stop poorly performing training runs early
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

# Get the training dataset
iris_ds = ws.datasets.get("iris-dataset")

# Create an estimator that uses the remote compute
hyper_estimator = SKLearn(source_directory=experiment_folder,
                           inputs=[iris_ds.as_named_input('iris-dataset')], # Pass the dataset as an input
                           compute_target = cpu_cluster,
                           conda_packages=['pandas','ipykernel','matplotlib'],
                           pip_packages=['azureml-sdk','argparse','pyarrow'],
                           entry_script='iris_training.py')

# Configure hyperdrive settings
hyperdrive = HyperDriveConfig(estimator=hyper_estimator, 
                          hyperparameter_sampling=params, 
                          policy=policy, 
                          primary_metric_name='Accuracy', 
                          primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                          max_total_runs=10,
                          max_concurrent_runs=4)


# Run the experiment
run = experiment.submit(config=hyperdrive)

#Get the best run 
best_run = run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details() ['runDefinition']['arguments']

print('Best Run Id: ', best_run.id)
print(' -Accuracy:', best_run_metrics['Accuracy'])
print(' -Maximum Tree Depth:',parameter_values)

#Register the best model 
best_run.register_model(model_path='outputs/iris_model.pkl', model_name='iris_model', tags={'Training context':'Hyperdrive'}, properties={'Accuracy': best_run_metrics['Accuracy']})

# List registered models
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')