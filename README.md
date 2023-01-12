# Demonstrating MATE and COCOA for Data Discovery

This repository contains the code of our data discovery system MaCO as well as the corresponding demonstration.

In [this video](https://youtu.be/cJfWn2wc_ZI), we demonstrate how the pipeline can be used to enrich a given input dataset with features from a data lake.

## Interactive demo in Google Colab
On [Google Colab](https://colab.research.google.com/github/LUH-DBS/datalake_indexes/blob/main/datalakes_indexes_demo.ipynb),
we provide the demonstration notebook. By running the code cells, MaCo is installed in the current session and can be used afterwards.

Please note: The duplicate tables graph cannot be displayed within Google Colab. Please use a local jupyter notebook instance for the full demo.

## Local Installation

1. Install [anaconda](https://www.anaconda.com/products/individual).
2. Create a new environment with python 3.9 and activate it.
```
conda create -n MaCo python=3.9
conda activate MaCo
```

3. Install the requirements via `pip install -r requirements.txt`.
4. Run a jupyter notebook instance `jupyter notebook`
5. Open `datalakes_indexes_demo.ipynb`

Code is tested on Mac only.

