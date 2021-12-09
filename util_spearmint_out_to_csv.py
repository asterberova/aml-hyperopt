import json
import glob
import numpy as np
import pandas as pd
import pickle
import csv
import os
import ast


if __name__ == "__main__":
    logreg = True
    if logreg: # if original benchmark was run
        path = 'spearmint_data/logreg/'
        output_path = 'csv_data/'
    else: # if surrogate benchmark was run
        path = 'spearmint_data/surrogate/'
        output_path = 'results/'

    runs = 10
    # runs = [1]
    for run in range(runs):
        print('run {}'.format(run))
        eval_dict = []
        run_path = os.path.join(path, 'run{}'.format(run))
        out_path = os.path.join(run_path, 'output')
        lst = os.listdir(out_path)
        sorted_list = sorted(lst)  # important to keep the order of evaluations
        not_include = 0
        for filename in sorted_list:
            # print(filename)
            file_path = os.path.join(out_path, filename)
            with open(file_path, 'r') as f:
                data = ast.literal_eval(f.read())
                if data in eval_dict and not_include < len(sorted_list)-100:    # preprocessing - keep only 100 samples
                    not_include += 1                                            # not including duplicates
                else:
                    eval_dict.append(data)

        df = pd.DataFrame.from_dict(eval_dict)
        print('Saved to {}'.format(os.path.join(output_path, 'spearmint{}.csv'.format(run))))
        df.to_csv(os.path.join(output_path, 'spearmint{}.csv'.format(run)), index=False, header=True)
