# Contents
The folder contains the following files and folders:
* **LightGBM_Classification.py**: The script to get the results below.
* **imbalanced_LGBM** folder - contains the prediction results using RF model trained on the original imbalance dataset; prediction, performance and importance of each CAZy Family.
* **smogn_LGBM** folder - contains the prediction results using RF model trained on the over-sampled dataset; prediction, performance and importance of each CAZy Family.

# Requirements
The experimental design was implemented in Python. In order to replicate these experiments you will need a working installation of Python.

You also need to install the following additional packages:
* pandas
* numpy
* scikit-learn
* smogn
* lightgbm
* optuna

All the above packages, with the exception of smogn package, is available at the Python Package Index and on Anaconda Cloud. smogn can be installed only from PyPI repository.

```
# PyPI
$ pip install package_name
```

```
# Anaconda Cloud
$ conda install -c conda-forge package_name
```