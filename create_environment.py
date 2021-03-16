# Import Libraries
from azureml.core import Environment,Workspace

# get workspace
ws = Workspace.from_config()

# Create environment
env = Environment.from_conda_specification(
        name='python-env',
        file_path='./.azureml/python-env.yml'
    )


# register environment
env.register(workspace=ws)
