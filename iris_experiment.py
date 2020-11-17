import os
import azureml.core
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

