# Contents
The folder contains the following files and folders:
* **RandomForest.py**: The script to get the results below.
* **imbalanced_RF** folder - contains the prediction results using RF model trained on the original imbalance dataset; prediction, performance and importance of each CAZy family.
* **smote_RF** folder - contains the prediction results using RF model trained on the over-sampled dataset; prediction, performance and importance of each CAZy family.

# Requirements
The experimental design was implemented in Python. In order to replicate these experiments you will need a working installation of Python.

You also need to install the following additional packages:
* pandas
* numpy
* scikit-learn
* imbalanced-learn

All the above packages is available at the Python Package Index and on Anaconda Cloud.

```
# PyPI
$ pip install package_name
```

```
# Anaconda Cloud
$ conda install -c conda-forge package_name
```