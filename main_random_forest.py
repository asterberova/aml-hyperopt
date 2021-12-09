from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from hyperopt.pyll import scope
from hyperopt import hp, tpe, rand, fmin, Trials, STATUS_OK
from hyperopt.pyll import scope

import json
import glob
import numpy as np
import pandas as pd
import pickle
import csv


########     LOAD CSV DATA      ########
# load all gathered csv data
path = 'csv_data'
filepath = glob.glob(path+'/*.csv')
# half data ofr training and second half for testing
# preprocessing - depends on how the data from each of us look like
data = pd.DataFrame()
for filename in filepath:
    print('Reading {}'.format(filename))
    df = pd.read_csv(filename)
    data = pd.concat((df, data), axis=0)
# print(data)

X = data.drop(columns=['loss'])
y = data['loss']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=33, shuffle=True)

print("Image Data Shape", X_train.shape)
print("Image test Data Shape", X_test.shape)
print("Target Data Shape", y_train.shape)
print("Target test Data Shape", y_test.shape)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def random_forest_acc(params):

    rf = RandomForestRegressor(min_samples_split=params['min_samples_split'],
                               n_estimators=params['n_estimators'],
                               max_features=params['max_features'])

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    # acc = rf.score(X_test, y_test)

    return rmse


def obj_func(params):

    err = random_forest_acc(params)

    return {'loss': err, 'status': STATUS_OK}


if __name__ == "__main__":
    """
    perform random search optimization on Random Forest Regressor
    to find best parameters for Random Forest as surrogate benchmark model
    """

    # define space for random search on Random Forest regression
    space = {
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 100, q=1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 2000, q=1)),
        'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2'])
    }

    trials = Trials()
    best_params = fmin(fn=obj_func,
                       space=space,
                       algo=rand.suggest,
                       max_evals=100,
                       trials=trials)

    print("Best parameters:", best_params)
    print(trials.best_trial['result']['loss'])

    if best_params['max_features'] == 0:
        best_params['max_features'] = 'auto'
    elif best_params['max_features'] == 1:
        best_params['max_features'] = 'sqrt'
    elif best_params['max_features'] == 2:
        best_params['max_features'] = 'log2'

    loss = trials.losses()
    val = trials.vals
    val['loss'] = loss
    # print(val)

    with open('random_forest_best_params.json', 'w') as f:
        f.write(json.dumps({"Loss": trials.best_trial['result']['loss'],
                            "Best params": best_params}, cls=NpEncoder))

    filename = 'random_forest_best_params.csv'
    header = ['min_samples_split', 'n_estimators', 'max_features', 'loss']
    values = (val.get(key, []) for key in header)
    data = (dict(zip(header, row)) for row in zip(*values))
    with open(filename, 'w') as f:
        wrtr = csv.DictWriter(f, fieldnames=header)
        wrtr.writeheader()
        wrtr.writerows(data)

