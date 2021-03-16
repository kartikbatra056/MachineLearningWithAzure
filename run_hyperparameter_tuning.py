# Import Libraries
from azureml.core import Experiment,ScriptRunConfig,Environment,Workspace
from azureml.train.hyperdrive import RandomParameterSampling,HyperDriveConfig,PrimaryMetricGoal,choice,MedianStoppingPolicy
from azureml.core.compute import ComputeTarget

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

# create run config
config = ScriptRunConfig(source_directory='src',
                         script='train_model.py',
                         arguments=['--n-estimator',10,
                                   '--min-samples-leaf',1,
                                   '--max-features','auto',
                                   '--input-data',ds.as_named_input('raw_data')],
                         environment=reg_env,
                         compute_target=compute_cluster)

# define hyper params
params = RandomParameterSampling({
        '--n-estimator': choice(10,100,150,200,250),
        '--min-samples-leaf':choice(1,2,3,4,5,6),
        '--max-features':choice('auto','sqrt','log2')
})

# policy
policy = MedianStoppingPolicy(evaluation_interval=1, delay_evaluation=3)

# hyperdrive configuration
hyperdrive = HyperDriveConfig(run_config=config,
                              hyperparameter_sampling=params,
                              policy=policy,
                              primary_metric_name='Testing F1 score',
                              primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                              max_total_runs=8,
                              max_concurrent_runs=4)

# Experiment run
experiment_name = 'hyper-drive-run'
experiment = Experiment(workspace=ws,name=experiment_name)
run = experiment.submit(config=hyperdrive)
run.wait_for_completion(show_output=True)
