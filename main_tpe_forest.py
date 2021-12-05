from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


from hyperopt.pyll import scope
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import json
import glob
import numpy as np
import pandas as pd
import pickle
import csv
from itertools import zip_longest


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
    # pre_data.append(df)
# data = pd.concat(pre_data, axis=0, ignore_index=True)
print(data)

X = data.drop(columns=['loss'])
y = data['loss']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=33, shuffle=True)

print("Image Data Shape", X_train.shape)
print("Image test Data Shape", X_test.shape)
print("Target Data Shape", y_train.shape)
print("Target test Data Shape", y_test.shape)


# define best founded parameters for random forest TODO:
best_params = {
    'min_samples_split': 1,
    'n_estimators': 1,
    'max_features': 1
}
# train Random Forrest regression with all gathered data
# and best founded parameters in previous random search
rf = RandomForestRegressor(min_samples_split=best_params['min_samples_split'],
                           n_estimators=best_params['n_estimators'],
                           max_features=best_params['max_features'])
rf.fit(X_train, y_train)


def random_forest_loss_predict(params):
    """
    Predict loss from trained Random Forest as surrogate benchmark model
    :param params:
    :return:
    """
    X_params = [params['lrate'], params['l2_reg'], params['n_epochs'] ]
    loss = rf.predict(X_params)
    return loss


# define space for TPE HPO
space = {"lrate": hp.uniform("lrate", 0, 1),
         "l2_reg": hp.uniform("l2_reg", 0, 1),
         # "batchsize": scope.int(hp.quniform("batchsize", 20, 2000, 1)),
         "n_epochs": scope.int(hp.quniform("n_epochs", 5, 2000, 1))}


def obj_func(params):
    """
    Objective function for TPE HPO
    :param params:
    :return: predicted loss from surrogate benchmark (random forest)
    """
    loss = random_forest_loss_predict(params)

    return {'loss': loss, 'status': STATUS_OK}


if __name__ == "__main__":
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
        print(val)

        # with open('best.json', 'w') as f:
        #     f.write(json.dumps({"Loss": trials.best_trial['result']['loss'],
        #                         "Best params": best_params}))

        filename = 'results/tpe{}.csv'.format(i)
        # header = ['lrate', 'l2_reg', 'batchsize', 'n_epochs', 'loss']
        header = ['lrate', 'l2_reg', 'n_epochs', 'loss']
        values = (val.get(key, []) for key in header)
        data = (dict(zip(header, row)) for row in zip(*values))
        with open(filename, 'w') as f:
            wrtr = csv.DictWriter(f, fieldnames=header)
            wrtr.writeheader()
            wrtr.writerows(data)



