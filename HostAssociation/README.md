# Contents
This folder has the code and dataset used in the experiments reported in the paper "Comparative genomics of wood-rot fungi by machine learning".

The folder is organized as follows:
* **RF_Classification** folder - contains the script and results of the experiment predicting host specialization label using Random Forest.
* **RF_Regression** folder - contains the script and results of the experiment predicting gymonosperm association value using Random Forest.
* **LightGBM_Classification** folder - contains the script and results of the experiment predicting host specialization label using LightGBM.
* **LightGBM_Regression** folder - contains the script and results of the experiment predicting gymonosperm association value using LightGBM.
* **data** folder - contains the dataset used in the above experiments.

# Requirements
The experimental design was implemented in Python. In order to replicate these experiments you will need a working installation of Python.

You also need to install the following additional packages:
* pandas
* numpy
* scikit-learn
* lightgbm
* imbalanced-learn
* optuna
* smogn

All the above packages, with the exception of smogn package, is available at the Python Package Index and on Anaconda Cloud. smogn can be installed only from PyPI repository.

```
# PyPI
$ pip install package_name
```

```
# Anaconda Cloud
$ conda install -c conda-forge package_name
```