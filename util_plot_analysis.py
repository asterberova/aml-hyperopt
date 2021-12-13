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
from sklearn.feature_selection import SelectKBest, chi2, f_classif


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
    # X_test, y_test = shuffle(X, y, random_state=33)
    #
    # print("Image Data Shape", X_test.shape)
    # print("Target Data Shape", y_test.shape)

    # return X_test, y_test
    return X, y


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


def plot_feature_vs_error_test(feature):

    plt.figure(figsize=(10, 6))

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

    # plt.scatter(y_test, y_pred)

    # plt.errorbar(X_test[feature], y_test, yerr=np.abs(y_test-y_pred), fmt='-')
    plt.scatter(X_test[feature], np.abs(y_test-y_pred))


    # plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.xlabel('Feature {}'.format(feature), fontsize=20)
    plt.ylabel('Loss difference', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.legend(fontsize=15)
    # plt.grid()
    # plt.xlim(limit)
    # plt.ylim(limit)
    plt.savefig('plots/feature_{}_error_test.png'.format(feature))
    plt.clf()
    print('Saved in plots/feature_{}_error_test.png'.format(feature))




def plot_feature_vs_error_looo(feature):

    plt.figure(figsize=(10, 7))

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

        # plt.scatter(y_test, y_pred, label=method.upper())
        # plt.scatter(y_test, y_pred)
        # plt.errorbar(X_test[feature], y_test, yerr=np.abs(y_test-y_pred), fmt='-')
        plt.scatter(X_test[feature], np.abs(y_test-y_pred), label=method.upper())


    # plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.xlabel('Feature {}'.format(feature), fontsize=25)
    plt.ylabel('Loss difference', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    plt.savefig('plots/feature_{}_error_looo.png'.format(feature))
    plt.clf()
    print('Saved in plots/feature_{}_error_looo.png'.format(feature))


# Examining the importance of features
def features_importance():
    # Load train data - all gathered csv
    X_train, y_train, X_test, y_test = load_csv_data_leave_ooo('real')
    features = [f for f in X_train]

    plt.figure(figsize=(6, 6))
    X_indices = np.arange(X_train.shape[-1])
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X_train, y_train)
    scores = selector.scores_
    plt.bar(X_indices, scores, width=.2)

    plt.xlabel("Feature", fontsize=15)
    plt.xticks(X_indices,features,rotation=90, fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Score", fontsize=15)
    # plt.grid()
    # plt.legend(fontsize=15)
    plt.savefig('plots/feature_importance.png', bbox_inches='tight')
    plt.clf()
    print('Saved in plots/feature_importance.png')


if __name__ == "__main__":
    # load data
    # X_train, y_train, X_test, y_test = load_csv_data_leave_ooo('real', 'spearmint')

    # X_train, y_train, X_test, y_test = load_csv_data_leave_ooo('real')
    # plot_performance_vs_prediction_looo([0.1, 0.20])
    # plot_performance_vs_prediction_looo([0, 1])
    #
    # plot_performance_vs_prediction_test([0.1, 0.20])
    # plot_performance_vs_prediction_test([0, 1])

    for feature in ['lrate', 'l2_reg', 'n_epochs']:
        # plot_feature_vs_error_test(feature)
        plot_feature_vs_error_looo(feature)

    # features_importance()

