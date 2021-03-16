# Import Libraries
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget,AmlCompute
from azureml.core.compute_target import ComputeTargetException

# get workspace
ws = Workspace.from_config()

# compute cluster name
cpu_cluster_name = 'kartik-cluster'

try:
    cpu_cluster = ComputeTarget(workspace=ws,name=cpu_cluster_name) # If cluster already exists
    print('Found Existing Cluster,use it.')

except ComputeTargetException:
    # compute cluster configuration
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2',
                                                           idle_seconds_before_scaledown=2400,
                                                           min_nodes=0,
                                                           max_nodes=2)
    # cluster creation
    cpu_cluster = ComputeTarget.create(ws,cpu_cluster_name,compute_config)

cpu_cluster.wait_for_completion(show_output=True)
