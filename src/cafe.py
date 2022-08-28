import pandas as pd
from data_handler import DataHandler
from cocoa import COCOA
from mate import MATE
import psycopg2
import logging


def run_cafe(dataset_path, query_columns):
    CONN_INFO = {
        'host': 'herkules.dbs.uni-hannover.de',
        'dbname': 'pdb',
        'user': 'jannis',
        'password': 'hgjasldhz',
    }

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    conn = psycopg2.connect(**CONN_INFO)
    data_handler = DataHandler(conn)

    # -------------------------------------------------------------------------------------------------------------------
    # ADDING DATASETS
    # -------------------------------------------------------------------------------------------------------------------
    # print(data_handler)
    # data_handler.add_tables_folder('/home/becktepe/datasets/opendata')
    # data_handler.add_tables_folder('./datasets')
    # data_handler.update_index()

    # -------------------------------------------------------------------------------------------------------------------
    # RUN MATE
    # -------------------------------------------------------------------------------------------------------------------
    mate = MATE(data_handler)
    input_dataset_name, input_dataset = data_handler.read_csv(dataset_path)
    input_dataset = input_dataset.head(5000)
    print(input_dataset.columns)
    top_joinable_tables = mate.enrich(input_dataset,
                                      query_columns,
                                      10,
                                      dataset_name=input_dataset_name)

    cocoa = COCOA(data_handler)
    print(cocoa.enrich_multicolumn(input_dataset, top_joinable_tables, 10, target_column='pf_ss_homicide'))
    # -------------------------------------------------------------------------------------------------------------------
    # RUN COCOA
    # -------------------------------------------------------------------------------------------------------------------

    # cocoa.enrich(input_data, 5, 100, query_column='director_name', target_column='imdb_score')

