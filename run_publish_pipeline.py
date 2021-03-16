# Import Libraries
from azureml.core import Experiment,Environment,Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData,Pipeline
from azureml.pipeline.steps import PythonScriptStep

# get workspace
ws = Workspace.from_config()

# get environment
reg_env = Environment.get(workspace=ws,name='python-env')

# get dataset
ds = ws.datasets.get('Fetal dataset')

# compute cluster name
cpu_cluster_name = 'kartik-cluster'

# get compute target
compute_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)

# pipeline run configuration
pipeline_config = RunConfiguration()

# use cluster created
pipeline_config.target = compute_cluster

# use environment created
pipeline_config.environment = reg_env

# temporary data store
prepare_datafolder = PipelineData('prepare_datafolder',datastore=ws.get_default_datastore())

# Step 1 run config
train_step = PythonScriptStep(name='Prepare data',
                              source_directory='src',
                              script_name='ps_1.py',
                              arguments=['--input-data',ds.as_named_input('raw_data'),
                                         '--prepared-data',prepare_datafolder],
                              outputs=[prepare_datafolder],
                              compute_target=compute_cluster,
                              runconfig=pipeline_config,
                              allow_reuse=True)

# Step 2 run config
register_step = PythonScriptStep(name='Train and register model',
                              source_directory='src',
                              script_name='ps_2.py',
                              arguments=['--prepared-data',prepare_datafolder],
                              inputs=[prepare_datafolder],
                              compute_target=compute_cluster,
                              runconfig=pipeline_config,
                              allow_reuse=True)

# get pipeline steps
pipeline_steps = [train_step,register_step]

# construct pipeline
pipeline = Pipeline(workspace=ws,steps=pipeline_steps)

# create experiment and run pipeline
experiment = Experiment(workspace=ws,name='Pipeline-run-experiment')
pipeline_run = experiment.submit(pipeline) # if wanna run complete pipeline again regenerate_outputs=True
pipeline_run.wait_for_completion(show_output=True)

# publish pipeline
pipeline_run.publish_pipeline(name='Training Pipeline',description='Trains fetal health prediction model',version='1.0')
