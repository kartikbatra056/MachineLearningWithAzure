# Import Libraries
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# get workspace
ws = Workspace.from_config()


# upload locally created wheel file to azure environment
whl_url = Environment.add_private_pip_wheel(workspace=ws,file_path ="src/dist/Myestimator-0.0.1-py3-none-any.whl")

# create CondaDependencies

myenv = CondaDependencies()
myenv.add_pip_package("scikit-learn")
myenv.add_pip_package("numpy")
myenv.add_pip_package("pandas")
myenv.add_pip_package("seaborn")
myenv.add_pip_package("matplotlib")
myenv.add_pip_package("azureml-defaults")
myenv.add_pip_package(whl_url)

# write dependencies to a yaml file
with open("./.azureml/python-env.yml","w") as f:
    f.write(myenv.serialize_to_string())

# Create environment
env = Environment.from_conda_specification(
        name='python-env',
        file_path='./.azureml/python-env.yml'
    )

# register environment
env.register(workspace=ws)
