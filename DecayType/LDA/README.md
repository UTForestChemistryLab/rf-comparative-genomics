# Contents
The folder contains the following files and folders:
* **LDA.py**: The script to get the results below.
* **imbalanced_LDA** folder - contains the prediction results using LDA model trained on the original imbalance dataset; prediction, performance and importance of each CAZy family.
* **smote_LDA** folder - contains the prediction results using LDA model trained on the over-sampled dataset; prediction, performance and importance of each CAZy family.
* **Families_for_Analysis.csv**: The list of CAZy families selected for building LDA models and their correlation ratios with decay type.
* **Correlated_Families.csv**: The list of CAZy families that were strongly correlated (r > 0.55) with the families selected above.

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