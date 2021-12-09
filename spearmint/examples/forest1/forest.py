import numpy as np
import pandas as pd
import sys
import math
import time
import glob
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle


# load all gathered csv data
# path = '../../../csv_data'
path = '/home/s2732815/aml-hyperopt/csv_data'
filepath = glob.glob(path+'/*.csv')
# half data ofr training and second half for testing
# preprocessing - depends on how the data from each of us look like
data = pd.DataFrame()
for filename in filepath:
    # print('Reading {}'.format(filename))
    df = pd.read_csv(filename)
    data = pd.concat((df, data), axis=0)

X = data.drop('loss', axis=1)
y = data['loss']
X_train, y_train = shuffle(X, y, random_state=33)

# Read best founded parameters for random forest
# f = open('../../../random_forest_best_params.json', 'r')
f = open('/home/s2732815/aml-hyperopt/random_forest_best_params.json', 'r')
param_json = json.loads(f.read())
best_params = param_json['Best params']

# Global Random Forest Regressor
# train Random Forrest regression with all gathered data
# and best founded parameters in previous random search
rf = RandomForestRegressor(min_samples_split=int(best_params['min_samples_split']),
                           n_estimators=int(best_params['n_estimators']),
                           max_features=best_params['max_features'])
rf.fit(X_train, y_train)


def random_forest_loss_predict(params):
    """
    Predict loss from trained Random Forest as surrogate benchmark model
    :param params:
    :return:
    """
    X_params = np.array([params['lrate'], params['l2_reg'], params['n_epochs']]).reshape(1, -1)
    loss = rf.predict(X_params)

    return loss[0]



# Write a function like this called 'main'
def main(job_id, params):
  # print 'Anything printed here will end up in the output directory for job #:', str(job_id)

  loss = random_forest_loss_predict(params)
  print({
      'lrate': params['lrate'][0],
      'l2_reg': params['l2_reg'][0],
      'n_epochs': params['n_epochs'][0],
      'loss': loss
  })

  return loss
