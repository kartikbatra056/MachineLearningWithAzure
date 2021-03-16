# Import Libraries
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from azureml.core import Run,Model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import joblib
import os

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--prepared-data',type=str,dest='prepared_folder',help='prepared dataset')
args = parser.parse_args()
path = args.prepared_folder

# Load train and test dataset
print('Loading data...')
train_df = pd.read_csv(os.path.join(path,'train.csv'))
test_df = pd.read_csv(os.path.join(path,'test.csv'))

# get run context
run = Run.get_context()

# seperate label column from dataset
X_train,y_train = train_df.drop(['fetal_health'],axis=1),train_df['fetal_health']
X_test,y_test = test_df.drop(['fetal_health'],axis=1),test_df['fetal_health']

# random forest model
lr = RandomForestClassifier(random_state=42)

# fit model
lr.fit(X_train,y_train)

y_pred = lr.predict(X_train)

# log training score
train_score = np.float(f1_score(y_train,y_pred,average='macro'))
run.log('Training F1 score',train_score)

cross_val = cross_val_score(lr,X_train,y_train,cv=5,scoring='f1_macro')

# log cross-val score
run.log('Mean cross val score',np.float(sum(cross_val)/len(cross_val)))

# log cross-val score list
run.log_list('Cross val scores list',cross_val)

y_pred = lr.predict(X_test)

# log testing score
test_score = np.float(f1_score(y_test,y_pred,average='macro'))
run.log('Testing F1 score',test_score)

# print classification report
print('Classification report \n',classification_report(y_test,y_pred))

# plot confusion matrix
fig = plt.figure(figsize=(7,5))
ax = sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)

# Log Confusion matrix
run.log_image('Confusion matrix',plot=fig)

# save model to output
os.makedirs('outputs',exist_ok=True)
joblib.dump(value=lr,filename='outputs/fetal_model.pkl')

# register model to workspace
print('Registering model...')
Model.register(workspace=run.experiment.workspace,
               model_path='outputs/fetal_model.pkl',
               model_name='Fetal-model-pipeline',
               tags={'Training_context':'Script'},
               properties={'Training Fscore':train_score,'Testing Fscore':test_score})
# complete run
run.complete()
