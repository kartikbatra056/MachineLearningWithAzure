# Import Libraries
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice,Webservice

# get workspace
ws = Workspace.from_config()

# set name to service
service_name = 'fetal-model-service'

# retrieve service
service = Webservice(name=service_name,workspace=ws)

print(service.state)

# delete web service
service.delete()
print('Service deleted.')
