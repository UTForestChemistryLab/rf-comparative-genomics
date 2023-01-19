# Contents
This folder has the code and dataset used in the experiments reported in the paper "Comparative genomics of wood-rot fungi by machine learning".

The folder is organized as follows:
* **SelectOverSampleTechnique** folder - contains the script and results of the experiment for the comparison of over-sampling techniques.
* **LDA** folder - contains the script and results of the Linear Discriminant Analysis decay type prediction experiment.
* **RandomForest** folder - contains the script and results of the Random Forest decay type prediction experiment.
* **data** folder - contains the dataset used in the above experiments.

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