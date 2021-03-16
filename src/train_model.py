# Import Libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from azureml.core import Run
from Myestimator import NewColumns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import joblib
import os

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n-estimator',type=int,default=100,dest='estimators',help='number of estimators to use.')
parser.add_argument('--max-depth',type=int,default=None,dest='max_depth',help='maximum depth of tree.')
parser.add_argument('--min-sample-split',type=int,default=2,dest='min_sample_split',help='minimum number of samples to have a split.')
parser.add_argument('--min-samples-leaf',type=int,default=1,dest='min_samples_leaf',help='minimum number of samples per leaf.')
parser.add_argument('--max-features',type=str,default='auto',choices=['auto','sqrt','log2'],dest='max_features',help='maximum number of features to use.')
parser.add_argument('--input-data',type=str,dest='dataset_folder',help='dataset')
args = parser.parse_args()

# get run context
run = Run.get_context()

# get hyperparameters
n_estimators = args.estimators
max_depth = args.max_depth
min_sample_split = args.min_sample_split
min_samples_leaf = args.min_samples_leaf
max_features = args.max_features

# log hyperparameters
run.log('n_estimators',n_estimators)
run.log('max_depth',max_depth)
run.log('min_samples_split',min_sample_split)
run.log('min_samples_leaf',min_samples_leaf)
run.log('max_features',max_features)

# Load dataset
print('Loading data...')
df = run.input_datasets['raw_data'].to_pandas_dataframe()

# perform train-test split
train_df,test_df = train_test_split(df,test_size=0.25,stratify=df['fetal_health'],random_state=1)

# seperate label column from dataset
X_train,y_train = train_df.drop(['fetal_health'],axis=1),train_df['fetal_health']
X_test,y_test = test_df.drop(['fetal_health'],axis=1),test_df['fetal_health']

# continuous columns
cont_cols = ['baseline value','uterine_contractions', 'abnormal_short_term_variability','mean_value_of_short_term_variability','percentage_of_time_with_abnormal_long_term_variability','mean_value_of_long_term_variability', 'histogram_width','histogram_min', 'histogram_max', 'histogram_mode', 'histogram_mean','histogram_median', 'histogram_variance','num_out','fetal_movement']

# categorical columns
cat_cols = ['accelerations','histogram_tendency','total_decelerations']

# params
params = {'n_estimators':n_estimators,
          'max_depth':max_depth,
          'min_samples_split':min_sample_split,
          'min_samples_leaf':min_samples_leaf,
          'max_features':max_features
          }

# preprocess columns
preprocess = ColumnTransformer(transformers=[('drop_columns','drop',['light_decelerations',
                                                                      'severe_decelerations',
                                                                      'prolongued_decelerations',
                                                                      'histogram_number_of_peaks',
                                                                      'histogram_number_of_zeroes']),
                                              ('scale_data',StandardScaler(),cont_cols),
                                              ('encode_data',OneHotEncoder(),cat_cols)])

# define pipeline steps
model_pipeline=Pipeline(steps=[('create_new_columns',NewColumns()),
                               ('preprocessing',preprocess),
                               ('random_forest',RandomForestClassifier(**params,random_state=1))])

# fit model pipeline
model_pipeline.fit(X_train,y_train)

# predict using training data
y_pred = model_pipeline.predict(X_train)

# log training score
run.log('Training F1 score',np.float(f1_score(y_train,y_pred,average='macro')))

cross_val = cross_val_score(model_pipeline,X_train,y_train,cv=5,scoring='f1_macro')

# log cross-val score
run.log('Mean cross val score',np.float(sum(cross_val)/len(cross_val)))

# log cross-val score list
run.log_list('Cross val scores list',cross_val)

y_pred = model_pipeline.predict(X_test)

# log testing score
run.log('Testing F1 score',np.float(f1_score(y_test,y_pred,average='macro')))

# print classification report
print('Classification report \n',classification_report(y_test,y_pred))

# plot confusion matrix
fig = plt.figure(figsize=(7,5))
ax = sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)

# Log Confusion matrix
run.log_image('Confusion matrix',plot=fig)

# save model to output
os.makedirs('outputs',exist_ok=True)
joblib.dump(value=model_pipeline,filename='outputs/fetal_model.pkl')

# complete run
run.complete()
