

# Installation
The three different HPO methods (SPEARMINT, SMAC, TPE) are implemented with different requirements, each using a different version of Python. The installation is divided into three parts.

### SPEARMINT
This package requires Python 2.7.
```bash
# create a new environment
# install dependencies
pip install -r requirements_spearmint.txt
cd spearmint/
bin/make_protobufs
python spearmint/setup.py install
```

###SMAC
This package requires Python 3.9.
```bash
# create a new environment
# install dependencies
pip install -r requirements_smac.txt
```

###TPE and rest of the scripts
This package requires Python 3.6.
```bash
# create a new environment
# install dependencies
pip install -r requirements.txt
```


# Running Experiments

### SPEARMINT on logistic regression on MNIST data
Performs SPEARMINT hyperparameter optimization of logistic regression on original MNIST dataset. Gathers the data of performance for each hyperparameters seting. 
```bash
# Run SPEARMINT HPO on logistic regression on MNIST data
cd spearmint/bin/
./spearmint ../examples/logreg1/config.pb --driver=local --method=GPEIOptChooser --method-args=noiseless=1
```

### SPEARMINT on surrogate benchmark model
Performs SPEARMINT hyperparameter optimization with trained surrogate benchmark model - which is Random Forest regressor. Random Forest Regressor is trained once at the beginning and prediction of loss from that model is used as objective function for SPEARMINT. Gathers loss per evaluation. 
```bash
# Run SPEARMINT HPO on surrogate
cd spearmint/bin/
./spearmint ../examples/forest1/config.pb --driver=local --method=GPEIOptChooser --method-args=noiseless=1
```


### SMAC on logistic regression on MNIST data
Performs SMAC hyperparameter optimization of logistic regression on original MNIST dataset. Gathers the data of performance for each hyperparameters seting. 
```bash
# Run SMAC HPO on logistic regression on MNIST data
python -m main_smac_logreg
```

### SMAC on surrogate benchmark model
Performs SMAC hyperparameter optimization with trained surrogate benchmark model - which is Random Forest regressor. Random Forest Regressor is trained once at the beginning and prediction of loss from that model is used as objective function for SPEARMINT. Gathers loss per evaluation. 
```bash
# Run SMAC HPO on surrogate
python -m main_smac_forest
```


### TPE on logistic regression on MNIST data
Performs TPE hyperparameter optimization of logistic regression on original MNIST dataset. Gathers the data of performance for each hyperparameters seting.
```bash
# Run TPE HPO on logistic regression on MNIST data
python -m main_tpe_logreg
```

### TPE on surrogate benchmark model
Performs TPE hyperparameter optimization with trained surrogate benchmark model - which is Random Forest regressor. Random Forest Regressor is trained once at the beginning and prediction of loss from that model is used as objective function for TPE. Gathers loss per evaluation.
```bash
# Run TPE HPO on surrogate
python -m main_tpe_forest
```


### Random search on logistic regression on MNIST data
Performs random search of hyperparameters of logistic regression on original MNIST dataset. Gathers the data of performance for each hyperparameters seting.
```bash
# Run andom search on logistic regression on MNIST data
python -m main_random_logreg
```

### Random search to find parameters for surrogate benchmark model
Performs random search of hyperparameters of Random Forest Regresor. "We used random search to optimize hyperparameters and considered 100 samples over the stated hyperparameters; we trained the model on 50% of the data, chose the best configuration based on its performance on the other 50%. Returns the best founded parameters."
```bash
# Run random search on random forest
python -m main_random_rofest
```

**cite** The implementation is based on:

    @inproceedings{paper,
	Author = {Katharina Eggensperger and Marius Lindauer and Holger H. Hoos and Frank Hutter and Kevin Leyton-Brown},
	Title = {Efficient benchmarking of algorithm configurators via model-based surrogates},
	Booktitle  = {Machine Learning},
	Year = {2017}
    }


