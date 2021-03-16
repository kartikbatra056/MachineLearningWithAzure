from sklearn.base import BaseEstimator

class NewColumns(BaseEstimator):

    def __init__(self):
        pass

    def fit(self,documents,y=None):
        return self

    def transform(self,data):

        data['total_decelerations']=(data['light_decelerations']+data['severe_decelerations']+data['prolongued_decelerations'] ==0.0)
        data['accelerations']=(data['accelerations']==0.0)
        data['num_out']=data['histogram_number_of_peaks']+data['histogram_number_of_zeroes']

        return data
