# Import Libraries
from azureml.core import Experiment,ScriptRunConfig,Environment,Workspace
from azureml.core.conda_dependencies import CondaDependencies

# get workspace
ws = Workspace.from_config()

# get dataset
ds = ws.datasets.get('Fetal dataset')

# create conda environment
env = Environment("env")
env.python.user_managed_dependencies = True

# create run config
config = ScriptRunConfig(source_directory='src',
                         script='train_model.py',
                         arguments=['--input-data',ds.as_named_input('raw_data')],
                         compute_target='local',
                         environment=env)

# Experiment run
experiment_name = 'local-training-run'
experiment = Experiment(workspace=ws,name=experiment_name)
run = experiment.submit(config=config)
run.wait_for_completion(show_output=True)

# register model to workspace
print('Registering model...')
model = run.register_model(model_path='outputs/fetal_model.pkl',model_name='fetal_model',
                  tags={'Training_context':'Script'},
                  properties={'Training Fscore':run.get_metrics()['Training F1 score'],'Testing Fscore':run.get_metrics()['Testing F1 score']})
