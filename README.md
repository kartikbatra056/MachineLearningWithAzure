# MachineLearningWithAzure
This is a Fetal health Prediction project built using Azure machine learning service.
Dataset used for project is freely available on Kaggle:[Fetal-Health-Prediction](https://www.kaggle.com/andrewmvd/fetal-health-classification)

For running the project one need to have an account on Azure-portal.
Following Project covers the steps involved in building a repeatable pipeline for building and deploying machine learning model.
* src - folder contains all scripts used to train and score model 
* data - folder contains dataset to be used for Our project 

Run given commands in bash terminal 

* Clone repository  ```git clone https://github.com/kartikbatra056/MachineLearningWithAzure.git```
* Enter cloned directory ```cd MachineLearningWithAzure```
* Create virtual environment ```python3 -m venv env``` 
* Install dependencies ```pip3 install -r requirements.txt``` 

Note - Be sure you are inside MachineLearningWithAzure Folder

# Create workspace
Now Before running file make some changes to file 
* create_workspace.py - Enter your Azure Subscription ID

Now one can easily create a workspace by running : ```python3 ./create_workspace.py```

# Create Environment
In the following project I have created a custom class for one of the preprocessing step which is converted to wheel file to be used for both development and testing environment.

* For creating environment - ```python3 ./create_environment.py```

Similarly you can,
* upload and register dataset - ```python3 ./upload_register_dataset.py```
* create a compute cluster -  ```python3 ./create_cluster.py```

# Training and register model  
Given file trains your model locally and save all other artifacts like model pickle file and its results on Azure for monitoring your model development.

* train and register model - ```python3 ./run_train_model.py```

run_train_model.py runs train_model.py file loacted in ```src``` folder.
train_model.py runs a scikit-learn pipeline which preprocesses data,trains model and saves complete pipeline to be used while inference and testing.

Note - Output of Above run is ```fetal-model.pkl``` which is Stored on Azure machine learning.

# Run Azure machine learning pipeline
Pipeline step 1 for preprocessing data- ```src/ps_1.py```
Pipeline step 2 for training model- ```src/ps_2.py```

* create and publish Azure machine learning pipeline - ```python3 ./run_publish_pipeline.py```

Above Code creates and publish a Azure machine learning Pipeline which preprocess data and save a trained model.
Once we have Published a pipeline we can run Pipeline whenever needed by just triggering published endpoint.

# Hyperparameter Tuning 
Following code runs hyperparameter tuning using Compute cluster that was created by running create_cluster.py so be sure run it before running below code 
* run hyperparameter tuning  - ```python3 ./run_hyperparameter_tuning.py```
Gives us best run out of different runs performed with Hyperparameter.

# Deploy endpoint

Now once we have model ready we can deploy it using ACI or AKS webservice on Azure .
For following run we will deploy it on ACI(Azure container Instance) 

* For deploying endpoint - ```python3 ./deploy_monitor_model.py```
* For deleting deployed service - ```python3 ./delete_deployed_service.py```

Aim of the Following project was to learn the steps involved in building and deploying a Machine learning model using Azure Machine learning service.
