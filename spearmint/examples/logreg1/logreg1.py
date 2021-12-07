import numpy as np
import sys
import math
import time
from mnist import MNIST
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import shuffle


# load MNIST dataset
# mndata = MNIST('python-mnist/data')
mndata = MNIST('/data/s2732815/python-mnist/data')

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()
y_train = np.array(y_train)
y_test = np.array(y_test)


def logreg(params):
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



# Write a function like this called 'main'
def main(job_id, params):
  # print 'Anything printed here will end up in the output directory for job #:', str(job_id)
  # print params
  # return branin(params['X'])
  # print params
  loss = logreg(params)
  print({
      'lrate': params['lrate'][0],
      'l2_reg': params['l2_reg'][0],
      'n_epochs': params['n_epochs'][0],
      'loss': loss
  })

  return loss
