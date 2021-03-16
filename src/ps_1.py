# Import Libraries
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from azureml.core import Run
import numpy as np
import pandas as pd
import argparse
import os

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input-data',type=str,dest='dataset_folder',help='dataset')
parser.add_argument('--prepared-data',type=str,dest='output_folder',help='output dataset')
args = parser.parse_args()

# output folder
save_folder = args.output_folder

# get run context
run = Run.get_context()

# Load dataset
print('Loading data...')
df = run.input_datasets['raw_data'].to_pandas_dataframe()

# preprocess dataset
df['accelerations'] = (df['accelerations']==0.0)
df['num_out'] = df['histogram_number_of_peaks']+df['histogram_number_of_zeroes']
df['total_decelerations'] = (df['light_decelerations']+df['severe_decelerations']+df['prolongued_decelerations'] ==0.0)

# drop unused columns
df.drop(['light_decelerations','severe_decelerations','prolongued_decelerations','histogram_number_of_peaks','histogram_number_of_zeroes'],axis=1,inplace=True)

# perform train-test split
train_df,test_df = train_test_split(df,test_size=0.25,stratify=df['fetal_health'])

# continuous columns
cont_cols = ['baseline value','uterine_contractions', 'abnormal_short_term_variability','mean_value_of_short_term_variability','percentage_of_time_with_abnormal_long_term_variability','mean_value_of_long_term_variability', 'histogram_width','histogram_min', 'histogram_max', 'histogram_mode', 'histogram_mean','histogram_median', 'histogram_variance','num_out','fetal_movement']

# Standard scale continuous values
scaler = StandardScaler()
train_df[cont_cols] = scaler.fit_transform(train_df[cont_cols])
test_df[cont_cols] = scaler.transform(test_df[cont_cols])

# categorical columns
cat_cols = ['accelerations','histogram_tendency','total_decelerations']

# Label encode categorical values
encoder = LabelEncoder()
for col in cat_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

# Create output folder
print('Saving output...')
os.makedirs(save_folder,exist_ok=True)
train_path = os.path.join(save_folder,'train.csv')  # train data path
test_path = os.path.join(save_folder,'test.csv')   # test data path

# Save training data
train_df.to_csv(train_path,index=False,header=True)

# Save testing data
test_df.to_csv(test_path,index=False,header=True)

# complete run
run.complete()
