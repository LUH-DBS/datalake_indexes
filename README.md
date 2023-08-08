# Demonstrating MATE and COCOA for Data Discovery

This repository contains the code of our data discovery system MaCO as well as the corresponding demonstration.

For the [WHO](https://youtu.be/5wOUBFbOQkw) and [Movie](https://youtu.be/QK02LqDNFAI) datasets, we demonstrate how the pipeline can be used to enrich a given input dataset with features from a data lake.

## Interactive demo in Google Colab
On [Google Colab](https://colab.research.google.com/github/LUH-DBS/datalake_indexes/blob/main/datalakes_indexes_demo.ipynb),
we provide the demonstration notebook. By running the code cells, MaCo is installed in the current session and can be used afterwards.

Please note: The duplicate tables graph cannot be displayed within Google Colab. Please use a local jupyter notebook instance for the full demo.

## Local Usage
Please follow these steps to use the demonstration notebook on your local machine:

1. Install [anaconda](https://www.anaconda.com/products/individual).
2. Create a new environment with python 3.9 and activate it.
```
conda create -n MaCo python=3.9
conda activate MaCo
```

3. Clone the GitHub repository
```
git clone https://github.com/LUH-DBS/datalake_indexes.git
cd datalake_indexes
```

4. Install the requirements via `pip install -r requirements.txt`.
5. Run a jupyter notebook instance `jupyter notebook`
6. Open `datalakes_indexes_demo.ipynb`

Code is tested on Mac only.

## MaCo installation
You can also install and use MaCo in your own projects by following these steps:

1. Install [anaconda](https://www.anaconda.com/products/individual).
2. Create a new environment with python 3.9 and activate it.
```
conda create -n MaCo python=3.9
conda activate MaCo
```

3. Clone the GitHub repository
```
git clone https://github.com/LUH-DBS/datalake_indexes.git
cd datalake_indexes
```

4. Install MaCo
```
pip install .
```

5. Use MATE and COCOA

```python
import pandas as pd
from maco.data_handler import DataHandler
from maco.cocoa import COCOA
from maco.mate import MATE
from maco.duplicate_detection import DuplicateDetection
from maco.util import get_cleaned_text
import psycopg2

conn = psycopg2.connect({
    "user": "user",
    "password": "password"
})

# Create a DataHandler and pass the DB relation names
data_handler = DataHandler(
    conn,
    main_table=f"MaCo_main_tokenized",
    column_headers_table=f"MaCo_column_headers",
    table_info_table=f"MaCo_table_info",
    cocoa_index_table=f"MaCo_cocoa_index"
)

# Index the data lake
data_handler.add_tables_folder("my_data_lake")  # add folder containing csv/parquet/json files
data_handler.update_index()

# Read and prepare the input dataset
input_dataset = pd.read_csv("input_dataset.csv")
input_dataset = input_dataset.applymap(get_cleaned_text)  # tokenization

# Joinability discovery using MATE
top_joinable_tables = MATE(data_handler).join_search(
    input_dataset,
    ["query_column_A", "query_column_B"],
    10,             # number of top table-column combinations to return 
    k_c=5000        # number of candidate table-column combinations to fetch
)

# Use the result
for joinability_score, table_id, columns, join_map in top_joinable_tables:
    pass
    
# Duplicate detection using XASH index
dup = DuplicateDetection(data_handler)

duplicate_tables = []      # stores all duplicate tables for the joinable tables

for _, table_id, _, _ in top_joinable_tables:
    table = data_handler.get_table(table_id)
    duplicate_tables += dup.get_duplicate_tables(table)

duplicate_relations = dup.get_relations(duplicate_tables)   # relations within duplicates

# Correlation calculation using COCOA
top_correlating_columns = COCOA(data_handler).enrich_multicolumn(
    input_dataset,
    top_joinable_tables,
    10,
    "target_column"
)

# Use the result
for corr_coeff, table_col_id, is_numeric in top_correlating_columns:
    pass

```
