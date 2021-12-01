from mnist import MNIST

from sklearn.linear_model import LogisticRegression, SGDClassifier

from hyperopt.pyll import scope
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import json
import numpy as np
import pickle
import csv
from itertools import zip_longest


# load MNIST dataset
mndata = MNIST('python-mnist/data')
# mndata = MNIST('/data/s2732815/python-mnist/data')

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()
y_train = np.array(y_train)
y_test = np.array(y_test)

print("Image Data Shape", len(X_train), len(X_train[0]))
print("Image test Data Shape", len(X_test), len(X_test[0]))
print("Target Data Shape", len(y_train))
print("Target test Data Shape", len(y_test))

# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')
# print("Image Data Shape" , mnist.data.shape)
# print("Label Data Shape", mnist.target.shape)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     mnist.data, mnist.target, test_size=1/7.0, random_state=0)



# define space for TPE HPO
space = {"lrate": hp.uniform("lrate", 0, 1),
         "l2_reg": hp.uniform("l2_reg", 0, 1),
         "batchsize": scope.int(hp.quniform("batchsize", 20, 2000, 1)),
         "n_epochs": scope.int(hp.quniform("n_epochs", 5, 2000, 1))}


def log_reg_acc(params):

    clf = SGDClassifier(loss='log',
                        learning_rate='constant',
                        eta0=params['lrate'],
                        penalty='elasticnet',
                        l1_ratio=params['l2_reg'])

    for n in range(params['n_epochs']):
        n_batches = len(X_train) // params['batchsize']
        for minibatch_idx in range(n_batches):
            # for minibatch_idx in range(1):
            clf.partial_fit(
                X_train[minibatch_idx * params['batchsize']: (minibatch_idx + 1) * params['batchsize']],
                y_train[minibatch_idx * params['batchsize']: (minibatch_idx + 1) * params['batchsize']],
                classes=np.unique(y_train))

    acc = clf.score(X_test, y_test)
    return acc


def obj_func(params):
    acc = log_reg_acc(params)
    loss = 1 - acc
    return {'loss': loss, 'status': STATUS_OK}


trials = Trials()
best_params = fmin(fn=obj_func,
                space=space,
                algo=tpe.suggest,
                max_evals=2,
                trials=trials)

print("Best parameters:",best_params)
print(trials.best_trial['result']['loss'])

# pickle.dump(trials, open("trials.json", "wb"))


loss = trials.losses()
val = trials.vals
val['loss'] = loss

print(val)
print(loss)

# arr_zip = zip(*(val['lrate'], val['l2_reg'], val['batchsize'], val['n_epochs'], loss))


with open('dominostats.json', 'w') as f:
    f.write(json.dumps({"Loss": trials.best_trial['result']['loss'],
                        "Best params" : best_params}))


filename = 'test.csv'
header = ['lrate', 'l2_reg', 'batchsize', 'n_epochs', 'loss']
values = (val.get(key, []) for key in header)
data = (dict(zip(header, row)) for row in zip(*values))
with open(filename, 'w') as f:
    #
    # writer = csv.writer(f)
    # writer.writerow(val.keys())
    # writer.writerows(zip(*val.values()))


    wrtr = csv.DictWriter(f, fieldnames=header)
    wrtr.writeheader()
    wrtr.writerows(data)



