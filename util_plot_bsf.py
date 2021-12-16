import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import ast

color_dict = {
    'spearmint': ['slateblue', 'mediumpurple', 'SPEARMINT', 'dashed'],
    'tpe': ['black', 'grey', 'TPE', 'dotted'],
    'smac': ['darkorange', 'lightcoral', 'SMAC', 'solid']
}


def array_to_bsf_values(arr):
    """
    Rewrite results to BSF values in time
    :param arr:
    :return: array
    """
    output = [arr[0]]
    for i in range(1, len(arr)):
        if output[i-1] < arr[i]:
            output.append(output[i-1])
        else:
            output.append(arr[i])

    return output


def compute_quartle(arr):
    """
    Compute quartile for each evaluation
    :param arr: 2D array
    :return: q1, q3
    """
    q1 = []
    q3 = []
    arr_t = np.transpose(arr)
    for line in arr_t:
        sorted_line = sorted(line)
        q1.append(np.median(sorted_line[:5]))
        q3.append(np.median(sorted_line[5:]))

    return q1, q3


def plot_bsf_median(benchmark):
    if benchmark == 'real':
        path = 'csv_data/'
    if benchmark == 'surrogate':
        path = 'results/'

    plt.figure(figsize=(10, 6))
    print('Collecting results...')
    for method in ['tpe', 'smac', 'spearmint']:
        losses = []
        for filename in os.listdir(path):
            if filename.startswith(method):
                file_path = os.path.join(path, filename)
                print('Reading {}'.format(file_path))
                data = pd.read_csv(file_path)
                # appends
                losses.append(array_to_bsf_values(data['loss']))

        median = np.median(losses, axis=0)
        q1, q3 = compute_quartle(losses)
        plt.plot(list(range(1, 101)), median, color=color_dict[method][0],
                 label=color_dict[method][2], linewidth=3, linestyle=color_dict[method][3])
        plt.fill_between(list(range(1, 101)), q1, q3, color=color_dict[method][1], alpha=0.4)

    plt.xlabel('Function evaluations', fontsize=18)
    plt.ylabel('Best loss achieved', fontsize=18)
    plt.xscale('log')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.xlim([1, 100])
    plt.ylim([0.09, 0.18])
    plt.savefig('plots/bsf_median_{}.png'.format(benchmark))
    plt.clf()
    print('Saved plots/bsf_median_{}.png'.format(benchmark))


def plot_bsf_mean(benchmark):
    if benchmark == 'real':
        path = 'csv_data/'
    if benchmark == 'surrogate':
        path = 'results/'

    min_loss = {
        'tpe': [],
        'spearmint':[],
        'smac':[]
    }

    plt.figure(figsize=(10, 6))
    print('Collecting results...')
    for method in ['tpe', 'smac', 'spearmint']:
        losses = []
        for filename in os.listdir(path):
            if filename.startswith(method):
                file_path = os.path.join(path, filename)
                print('Reading {}'.format(file_path))
                data = pd.read_csv(file_path)
                min_loss[method].append(min(data['loss']))
                # appends
                losses.append(array_to_bsf_values(data['loss']))
        # computing mean
        mean = np.mean(losses, axis=0)
        # print('MEAN')
        # print(mean)
        # computing std for confidence interval as std/sqrt(n)
        std = np.std(losses, axis=0) / np.sqrt(len(losses))
        # print('STD')
        # print(mean-std)
        # print(mean+std)

        plt.plot(list(range(1, 101)), mean, color=color_dict[method][0],
                 label=color_dict[method][2], linewidth=3, linestyle=color_dict[method][3])
        plt.fill_between(list(range(1, 101)), mean-std, mean+std, color=color_dict[method][1], alpha=0.4)

    plt.xlabel('Function evaluations', fontsize=18)
    plt.ylabel('Best loss achieved', fontsize=18)
    plt.xscale('log')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.xlim([1, 100])
    plt.savefig('plots/bsf_mean_{}.png'.format(benchmark))
    plt.clf()
    print('Saved plots/bsf_mean_{}.png'.format(benchmark))

    return min_loss



if __name__ == "__main__":
    # Plot meddian values and quartile
    plot_bsf_median('real')
    plot_bsf_median('surrogate')

    # Plot mean values and std
    min_loss_real = plot_bsf_mean('real')
    min_loss_surr = plot_bsf_mean('surrogate')

    print('REAL BENCHMARK:')
    for key in min_loss_real.keys():
        print('{} : {}+-{}'.format(key, np.mean(min_loss_real[key]), np.std(min_loss_real[key])))

    print('SURROGATE BENCHMARK:')
    for key in min_loss_surr.keys():
        print('{} : {}+-{}'.format(key, np.mean(min_loss_surr[key]), np.std(min_loss_surr[key])))

