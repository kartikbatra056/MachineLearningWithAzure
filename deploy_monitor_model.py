# Import Libraries
from azureml.core import Workspace
from azureml.core.model import Model,InferenceConfig
from azureml.core.webservice import AciWebservice,Webservice
import joblib
import json
import requests

# get workspace
ws = Workspace.from_config()

# set name to service
service_name = 'fetal-model-service'

# set script_file path
script_file = 'src/score_model.py'

# set environment path
env_file ='.azureml/python-env.yml'

# configure scoring environment
inference_config = InferenceConfig(runtime='python',
                                   entry_script=script_file,
                                   conda_file=env_file)

model = Model(workspace=ws,name='fetal_model') # get model

# set deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1,memory_gb =1)

# deploy service
aci_service = Model.deploy(workspace =ws,
                           name=service_name,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=deployment_config)

aci_service.wait_for_deployment(show_output=True)
print(aci_service.state)

# enable application insights for real time model monitoring
aci_service.update(enable_app_insights=True)

# get endpoint
endpoint = aci_service.scoring_uri

# new input
x_new = [[142,0.002,0.002,0.008,0,0,0,74,0.4,36,5,42,117,159,2,1,145,143,145,1,0]]

# input json
input_json = json.dumps({'data':x_new})

# set headers
headers = {'Content-type':'application/json'}

# get predictions
predictions = requests.post(endpoint,input_json,headers=headers)
print('Status:',predictions.status_code)
print(predictions)

if predictions.status_code == 200:
    predicted_class = json.loads(predictions.json())
    for i in range(len(x_new)):
        print ("Patient is:",predicted_class[i])
