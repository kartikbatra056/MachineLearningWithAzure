# load Libraries
from azureml.core import Workspace

# Create Workspace
ws = Workspace.create(name='KartikWorkspace',
                      subscription_id='Your-subscription-id',
                      resource_group='KartikResource',
                      create_resource_group=True,
                      location='eastus',
                      )

# save workspace configuration as file
ws.write_config(path='.azureml')
print('Workspace created.')
