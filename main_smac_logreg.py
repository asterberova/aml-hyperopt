from mnist import MNIST

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import shuffle

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory


import json
import numpy as np
import pickle
import csv
from itertools import zip_longest


# load MNIST dataset
# mndata = MNIST('python-mnist/data')
mndata = MNIST('/data/s2732815/python-mnist/data')

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()
y_train = np.array(y_train)
y_test = np.array(y_test)

print("Image Data Shape", len(X_train), len(X_train[0]))
print("Image test Data Shape", len(X_test), len(X_test[0]))
print("Target Data Shape", len(y_train))
print("Target test Data Shape", len(y_test))



def log_reg_loss(params):
    """
    Train SGD for logistic regression with chosen parameters
    :param params:
    :return: loss
    """

    clf = SGDClassifier(loss='log',
                        learning_rate='constant',
                        eta0=params['lrate'],
                        penalty='elasticnet',
                        l1_ratio=params['l2_reg'],
                        max_iter=params['n_epochs'],
                        shuffle=True)

    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    loss = 1 - acc

    return loss



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
        # "wallclock-limit": 100,
    })


    num_repeat = 10
    for i in range(num_repeat):
        print(f'Run {i}/{num_repeat}')
        # perform SMAC optimization and do logging
        # smac = SMAC4BB(scenario=scenario, tae_runner=log_reg_loss)
        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(i),
                        # intensifier_kwargs=intensifier_kwargs,
                        tae_runner=log_reg_loss)

        best_params = smac.optimize()

        rh = smac.runhistory
        # print(rh)
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

        # print("Best parameters:",best_params)
        # print(trials.best_trial['result']['loss'])
        #
        # loss = trials.losses()
        # val = trials.vals
        # val['loss'] = loss
        # print(val)

        # with open('best.json', 'w') as f:
        #     f.write(json.dumps({"Loss": trials.best_trial['result']['loss'],
        #                         "Best params": best_params}))

        filename = 'csv_data/smac{}.csv'.format(i)
        # header = ['lrate', 'l2_reg', 'batchsize', 'n_epochs', 'loss']
        header = ['lrate', 'l2_reg', 'n_epochs', 'loss']
        values = (val.get(key, []) for key in header)
        data = (dict(zip(header, row)) for row in zip(*values))
        with open(filename, 'w') as f:
            wrtr = csv.DictWriter(f, fieldnames=header)
            wrtr.writeheader()
            wrtr.writerows(data)



