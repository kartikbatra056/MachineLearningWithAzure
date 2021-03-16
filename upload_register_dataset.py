# A workaround to deal with .NET runtime not available error
from dotnetcore2 import runtime
runtime.version = ("18", "10", "0")
# load Libraries
from azureml.core import Workspace
from azureml.core import Dataset

# get workspace
ws = Workspace.from_config()

# get default datastore
default_ds = ws.get_default_datastore()

# upload data to default datastore
default_ds.upload_files(files=['./data/fetal_health.csv'],
                        target_path='Fetal-data/',
                        overwrite=True,
                        show_progress=True)

# create dataset
tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds,'Fetal-data/*.csv'))

# register tabular dataset
try:
    tab_data_set = tab_data_set.register(workspace=ws,
                                         name='Fetal dataset',
                                         description='Fetal Health Prediction',
                                         tags={'format':'csv'},
                                         create_new_version=True)

except Exception as ex:
        print(ex)

print('Dataset Registered')
