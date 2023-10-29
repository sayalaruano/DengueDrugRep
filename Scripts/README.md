# **Structure of the folder**
The scripts in this folder are used to train and evaluate the KGNN models. 

Before running the evaluation scripts, you should download the trained models from [Zenodo][models_doi] and save them in a folder called `Models/`.

The scripts are organized as follows:

|File|Description|
|:-:|---|
|[Training_KGNN_models_Pykeen.ipynb](Training_KGNN_models_Pykeen.ipynb)|Jupyter notebook to train the KGNN models|
|[Internal_performance_evaluation_KGNNs.py](Internal_performance_evaluation_KGNNs.py)|Python script to perform the internal evaluation of the trained KGNN models|
|[External_performance_evaluation_KGNNs.py](External_performance_evaluation_KGNNs.py)|Python script to perform the external evaluation of the trained KGNN models|
|[EDA_DRKG_compounds_names.py](EDA_DRKG_compounds_names.py)|Python script to do exploratory data analysis, and obtain the IDs and datasource of the compounds in the DRKG|

[models_doi]: https://zenodo.org/doi/10.5281/zenodo.10010151