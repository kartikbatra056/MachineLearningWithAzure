from azureml.core import Model
import json
import joblib
import numpy as np
import pandas as pd

# initialize model
def init():
    global model

    model_path = Model.get_model_path('fetal_model') # get model path
    model = joblib.load(model_path) # get model

def run(raw_data):

        data = np.array(json.loads(raw_data)['data']) # load data into numpy array

        data = np.nan_to_num(data) # replace nan with 0.0

        # input columns
        columns = ['baseline value', 'accelerations', 'fetal_movement',
                    'uterine_contractions', 'light_decelerations', 'severe_decelerations',
                    'prolongued_decelerations', 'abnormal_short_term_variability',
                    'mean_value_of_short_term_variability',
                    'percentage_of_time_with_abnormal_long_term_variability',
                    'mean_value_of_long_term_variability', 'histogram_width',
                    'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
                    'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
                    'histogram_median', 'histogram_variance', 'histogram_tendency']

        # define input dataframe
        input_df = pd.DataFrame(data,columns=columns)

        # get model prediction
        pred = model.predict(input_df)

        # get labels
        labels = {
                    1.0:'Normal',
                    2.0:'Suspect',
                    3.0:'Pathological'
                 }
        #list of final predictions
        predictions = []

        # convert predictions to labels
        for p in pred:
            predictions.append(labels[p])

        return json.dumps(predictions)
