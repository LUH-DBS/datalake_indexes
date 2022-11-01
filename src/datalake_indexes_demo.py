import numpy as np
import pandas as pd
from data_handler import DataHandler
from cocoa import COCOA
from mate import MATE
from duplicate_detection import DuplicateDetection
import psycopg2
import logging
import json
from typing import List, Tuple
from pyvis.network import Network


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

        self.__input_dataset = None
        self.__query_columns = None
        self.__target_column = None

        self.__tables_dict: dict = {}     # stores tables, key is tableId
        self.__top_joinable_tables: List[Tuple[int, int, List[int], np.ndarray]] = []

    def read_input(self, path: str) -> None:
        """
        Reads and stores an input dataset from csv.

        Parameters
        ----------
        path : str
            Path to csv file.
        """
        self.__input_dataset, _ = self.__data_handler.read_csv(path)

    def set_query_columns(self, query_columns: List[str]) -> None:
        """
        Sets the query columns that will be used for joinability discovery.

        Parameters
        ----------
        query_columns : List[str]
            List of input dataset columns.
        """
        if self.__input_dataset is None:
            print("Please load input dataset first.")
            return

        for query_column in query_columns:
            if query_column not in self.__input_dataset.columns:
                print(f"{query_column} not in input dataset.")
                return

        self.__query_columns = query_columns

    def set_target_column(self, target_column: str):
        """
        Sets the target column that will be used for correlation calculation.

        Parameters
        ----------
        target_column: str
            Input dataset column.
        """
        if self.__input_dataset is None:
            print("Please load input dataset first.")
            return

        if target_column not in self.__input_dataset.columns:
            print(f"{target_column} not in input dataset.")
            return

        self.__target_column = target_column

    def joinability_discovery(
            self,
            k: int = 10
    ) -> None:
        """
        Finds joinable tables within the datalake using MATE.

        Parameters
        ----------
        k : int
            Number of candidates that will be returned.
        """
        mate = MATE(self.__data_handler)
        self.__top_joinable_tables = mate.join_search(self.__input_dataset,
                                                      self.__query_columns,
                                                      k)

        joinable_columns_dict = {}
        column_headers_dict = {}
        for score, table_id, columns, join_map in self.__top_joinable_tables:
            joinable_columns_dict[table_id] = columns

            try:
                table = self.__data_handler.get_table(table_id)
            except:
                continue
            self.__tables_dict[table_id] = table

            column_headers = [table.columns[int(col_id)] for col_id in columns.split('_')][
                             :len(self.__query_columns)]
            column_headers_dict[table_id] = column_headers

    def display_joinable_table(self, rank: int) -> None:
        if rank < 0 or rank >= len(self.__top_joinable_tables):
            print(f"Invalid rank: {rank}")
            return

        score, table_id, columns, _ = self.__top_joinable_tables[rank]

        print(f"Score: {score},"
              f"Table ID: {table_id},"
              f"Joinable columns: {columns},"
              f"#rows: {self.__tables_dict[table_id].shape[0]},"
              f"#columns: {self.__tables_dict[table_id].shape[1]}")

        # TODO display table

        #highlight_sample = highlight_columns(tables_dict[table_id], column_headers_dict[table_id])
        #display(HTML(highlight_sample.to_html()))

    def duplicate_detection(self):
        dup = DuplicateDetection(self.__data_handler)

        #duplicate_detection_start = time.time()
        duplicate_tables = []
        for _, table_id, _, _ in self.__top_joinable_tables:
            table = self.__tables_dict[table_id]
            duplicate_tables += dup.get_duplicate_tables(table)

        duplicate_relations = dup.get_relations(duplicate_tables)

        net = Network(height='1000px', width='100%', notebook=True)

        for t in duplicate_relations:
            net.add_node(t[0], str(t[0]))
            net.add_node(t[1], str(t[1]))
            net.add_edge(t[0], t[1])

        # net.add_node(0,"0")
        # for t in duplicate_tables_first:
        #    net.add_node(t,str(t))
        #    net.add_edge(0, t)

        net.show_buttons(filter_=['physics'])
        net.set_edge_smooth("dynamic")
        net.toggle_stabilization(False)
        net.toggle_physics(False)

        # Get row values to generate html tables:
        output = ""
        for table_id in duplicate_tables:
            # print(data_handler.get_table(table_id).head(10).to_html())
            output += self.__data_handler.get_table(table_id).to_html(table_id=f"t{table_id}",
                                                                      index=None)

        # Convert CSV table to html table
        output = output + table.iloc[:10, :].to_html(table_id='t0', index=None)

        with open("template.html", 'r') as file:
            filedata = file.read()

        # Replace table placeholder with actual tables html
        filedata = filedata.replace('%%tables_placeholder%%', output)

        with open("template_new.html", 'w') as file:
            file.write(filedata)

        net.prep_notebook(custom_template=True, custom_template_path="template_new.html")

    def compare_XASH_alternations(self):
        pass

    def correlation_calculation(self):
        pass

    def fit_and_evaluate_moodel(self):
        pass

