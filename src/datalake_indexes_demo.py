import pandas as pd
from data_handler import DataHandler
from cocoa import COCOA
from mate import MATE
import psycopg2
import logging
import json


class DatalakeIndexesDemo:
    """
    Provides all required tools to run the datalake indexes demonstration scenario.
    """
    def __init__(
            self,
            config_path: str,
            datalake: str
    ):
        db_config = json.load(open(config_path))
        conn = psycopg2.connect(**db_config)

        self.__data_handler = DataHandler(
            conn,
            main_table=f"{datalake}_main_tokenized",
            column_headers_table=f"{datalake}_column_headers",
            table_info_table=f"{datalake}_table_info",
            cocoa_index_table=f"{datalake}_cocoa_index"
        )

    def joinability_discovery(self):
        pass

    def duplicate_detection(self):
        pass

    def compare_XASH_alternations(self):
        pass

    def correlation_calculation(self):
        pass

    def fit_and_evaluate_moodel(self):
        pass

