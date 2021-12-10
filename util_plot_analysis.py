import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import ast
# sklearn imports
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor


def load_csv_data_leave_ooo(benchmark, o_out=None):
    if benchmark == 'real':
        path = 'csv_data/'
    if benchmark == 'surrogate':
        path = 'results/'

    methods = ['tpe', 'smac', 'spearmint', 'random']
    if o_out:
        methods.remove(o_out)

    data = pd.DataFrame()
    for method in methods:
        for filename in os.listdir(path):
            if filename.startswith(method):
                file_path = os.path.join(path, filename)
                print('Reading {}'.format(file_path))
                df = pd.read_csv(file_path)
                data = pd.concat((df, data), axis=0)

    X = data.drop(columns=['loss'])
    y = data['loss']
    X_train, y_train = shuffle(X, y, random_state=33)

    print("Image Data Shape", X_train.shape)
    print("Target Data Shape", y_train.shape)

    data_out = pd.DataFrame()
    if o_out:
        for filename in os.listdir(path):
            if filename.startswith(o_out):
                file_path = os.path.join(path, filename)
                print('Reading {}'.format(file_path))
                df = pd.read_csv(file_path)
                data_out = pd.concat((df, data_out), axis=0)
        X = data_out.drop(columns=['loss'])
        y = data_out['loss']
        X_test, y_test = shuffle(X, y, random_state=33)

        print("Image test Data Shape", X_test.shape)
        print("Target test Data Shape", y_test.shape)
    else:
        X_test, y_test = [], []

    return X_train, y_train, X_test, y_test

def load_test_data():
    path = 'test_data/'

    data = pd.DataFrame()
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        print('Reading {}'.format(file_path))
        df = pd.read_csv(file_path)
        data = pd.concat((df, data), axis=0)

    X = data.drop(columns=['loss'])
    y = data['loss']
    X_test, y_test = shuffle(X, y, random_state=33)

    print("Image Data Shape", X_test.shape)
    print("Target Data Shape", y_test.shape)

    return X_test, y_test


def plot_performance_vs_prediction_looo(limit):

    plt.figure(figsize=(8, 8))

    for method in ['tpe', 'smac', 'spearmint']:
        # Load train and test data
        X_train, y_train, X_test, y_test = load_csv_data_leave_ooo('real', method)
        # Read best founded parameters for random forest
        f = open('random_forest_best_params.json', 'r')
        param_json = json.loads(f.read())
        best_params = param_json['Best params']
        # Train Random Forest
        rf = RandomForestRegressor(min_samples_split=int(best_params['min_samples_split']),
                                   n_estimators=int(best_params['n_estimators']),
                                   max_features=best_params['max_features'])
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        plt.scatter(y_test, y_pred, label=method.upper())

    plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.xlabel('True loss', fontsize=20)
    plt.ylabel('Predicted loss', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.xlim(limit)
    plt.ylim(limit)
    plt.savefig('plots/scatter_looo_{}-{}.png'.format(limit[0], limit[1]))
    plt.clf()
    print('Saved in plots/scatter_looo_{}-{}.png'.format(limit[0], limit[1]))


def plot_performance_vs_prediction_test(limit):

    plt.figure(figsize=(8, 8))

    # Load train data - all gathered csv
    X_train, y_train, X_test, y_test = load_csv_data_leave_ooo('real')
    # Load test data
    X_test, y_test = load_test_data()
    # Read best founded parameters for random forest
    f = open('random_forest_best_params.json', 'r')
    param_json = json.loads(f.read())
    best_params = param_json['Best params']
    # Train Random Forest
    rf = RandomForestRegressor(min_samples_split=int(best_params['min_samples_split']),
                               n_estimators=int(best_params['n_estimators']),
                               max_features=best_params['max_features'])
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    plt.scatter(y_test, y_pred)

    plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.xlabel('True loss', fontsize=20)
    plt.ylabel('Predicted loss', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.legend(fontsize=15)
    plt.grid()
    plt.xlim(limit)
    plt.ylim(limit)
    plt.savefig('plots/scatter_test_data_{}-{}.png'.format(limit[0], limit[1]))
    plt.clf()
    print('Saved in plots/scatter_test_data_{}-{}.png'.format(limit[0], limit[1]))


if __name__ == "__main__":
    # load data
    # X_train, y_train, X_test, y_test = load_csv_data_leave_ooo('real', 'spearmint')

    # X_train, y_train, X_test, y_test = load_csv_data_leave_ooo('real')
    plot_performance_vs_prediction_looo([0.1, 0.20])
    plot_performance_vs_prediction_looo([0, 1])

    plot_performance_vs_prediction_test([0.1, 0.20])
    plot_performance_vs_prediction_test([0, 1])
