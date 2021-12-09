# SMAC imports
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory

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


# Read best founded parameters for random forest
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
    :return: loss
    """
    X_params = np.array([params['lrate'], params['l2_reg'], params['n_epochs']]).reshape(1, -1)
    loss = rf.predict(X_params)

    return loss[0]




if __name__ == "__main__":

    param_space = ConfigurationSpace()

    lrate = UniformFloatHyperparameter('lrate', 0, 1)
    l2_rate = UniformFloatHyperparameter('l2_reg', 0, 1)
    n_epochs = UniformIntegerHyperparameter('n_epochs', 5, 2000)

    param_space.add_hyperparameters([lrate, l2_rate, n_epochs])

    scenario = Scenario({
        'run_obj': 'quality',
        'cs': param_space,
        'runcount-limit': 100,
        'deterministic': True,
    })


    num_repeat = 10
    for i in range(num_repeat):
        print(f'Run {i}/{num_repeat}')
        # perform SMAC optimization and do logging
        # smac = SMAC4BB(scenario=scenario, tae_runner=log_reg_loss)
        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(i),
                        # intensifier_kwargs=intensifier_kwargs,
                        tae_runner=random_forest_loss_predict)

        best_val = smac.optimize()

        rh = smac.runhistory
        val = {
            'lrate': [],
            'l2_reg': [],
            'n_epochs': [],
            'loss': []
        }
        confs = rh.get_all_configs()
        for c in confs:
            loss = rh.get_cost(c)
            for key in val.keys():
                if key == 'loss':
                    val[key].append(loss)
                else:
                    val[key].append(c._values[key])


        filename = 'results/smac{}.csv'.format(i)
        header = ['lrate', 'l2_reg', 'n_epochs', 'loss']
        values = (val.get(key, []) for key in header)
        data = (dict(zip(header, row)) for row in zip(*values))
        with open(filename, 'w') as f:
            wrtr = csv.DictWriter(f, fieldnames=header)
            wrtr.writeheader()
            wrtr.writerows(data)



