# Comparative genomics of wood-rot fungi by machine learning
This repository has all the code and datasets used in the experiments carried out in the paper "Comparative genomics of wood-rot fungi by machine learning".

The repository is organized into the following two folders:
* **DecayType** folder - contains all the code and datasets for reproducing the experiments on decay-type.
* ~~**HostAssociation** folder - contains all the code and datasets for reproducing the experiments on host specialization.~~ This code is still private.

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
