from datalake_indexes.data_handler import DataHandler
import json
import psycopg2
import logging
import os

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

DATALAKE = "gittables"
CONFIG_PATH = "/home/becktepe/db_config.json"
ROOT_DIR = '/home/becktepe/datasets/gittables_1M'


db_config = json.load(open(CONFIG_PATH))
conn = psycopg2.connect(**db_config)

data_handler = DataHandler(
    conn,
    main_table=f"{DATALAKE}_main_tokenized",
    column_headers_table=f"{DATALAKE}_column_headers",
    table_info_table=f"{DATALAKE}_table_info",
    cocoa_index_table=f"{DATALAKE}_cocoa_index"
)

print(data_handler)

data_handler.clean_up_db()
for subdir, _, _ in os.walk(ROOT_DIR):
    if subdir == ROOT_DIR:
        continue
    data_handler.add_tables_folder(subdir)

data_handler.update_index()
