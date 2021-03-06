# Hyperopt imports
from hyperopt.pyll import scope
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

# sklearn imports
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor

import json
import glob
import numpy as np
import pandas as pd
import csv


# load all gathered csv data
path = 'csv_data'
filepath = glob.glob(path+'/*.csv')
# half data ofr training and second half for testing
# preprocessing - depends on how the data from each of us look like
data = pd.DataFrame()
for filename in filepath:
    # print('Reading {}'.format(filename))
    df = pd.read_csv(filename)
    data = pd.concat((df, data), axis=0)
# print(data)

X = data.drop(columns=['loss'])
y = data['loss']
X_train, y_train = shuffle(X, y, random_state=33)

print("Image Data Shape", X_train.shape)
print("Target Data Shape", y_train.shape)


# Read best founded parameters for random forest TODO:
f = open('random_forest_best_params.json', 'r')
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

    return loss


def obj_func(params):
    """
    Objective function for TPE HPO
    :param params:
    :return: predicted loss from surrogate benchmark (random forest)
    """
    loss = random_forest_loss_predict(params)

    return {'loss': loss, 'status': STATUS_OK}


if __name__ == "__main__":

    # define space for TPE HPO
    space = {"lrate": hp.uniform("lrate", 0, 1),
             "l2_reg": hp.uniform("l2_reg", 0, 1),
             "n_epochs": scope.int(hp.quniform("n_epochs", 5, 2000, 1))}

    num_repeat = 10

    for i in range(num_repeat):
        print(f'Run {i}/{num_repeat}')
        # perform TPE optimization and do logging
        trials = Trials()
        best_params = fmin(fn=obj_func,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=100,
                        trials=trials)

        print("Best parameters:", best_params)
        print(trials.best_trial['result']['loss'])

        loss = trials.losses()
        val = trials.vals
        val['loss'] = loss
        # print(val)


        filename = 'results/tpe{}.csv'.format(i)
        header = ['lrate', 'l2_reg', 'n_epochs', 'loss']
        values = (val.get(key, []) for key in header)
        data = (dict(zip(header, row)) for row in zip(*values))
        with open(filename, 'w') as f:
            wrtr = csv.DictWriter(f, fieldnames=header)
            wrtr.writeheader()
            wrtr.writerows(data)



