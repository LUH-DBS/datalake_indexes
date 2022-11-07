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
from util import get_cleaned_text, generate_XASH
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.__output_dataset = None
        self.__external_columns = []

        self.__tables_dict: dict = {}     # stores tables, key is tableId
        self.__joinable_columns_dict: dict = {}
        self.__column_headers_dict: dict = {}
        self.__top_joinable_tables: List[Tuple[int, int, List[int], np.ndarray]] = []

        self.__top_correlating_columns: List = [Tuple[float, str]]
        self.__spearman_dict = {}
        self.__pearson_dict = {}

    def read_input(self, path: str) -> None:
        """
        Reads and stores an input dataset from csv.

        Parameters
        ----------
        path : str
            Path to csv file.
        """
        _, self.__input_dataset = self.__data_handler.read_csv(path)
        # TODO display input dataset
        print(self.__input_dataset.head())

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

        # TODO display input dataset with highlighted query columns
        print(self.__query_columns)
        print(self.__input_dataset.head())

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

        # TODO display input dataset with highlighted query and target columns
        print(self.__target_column)
        print(self.__input_dataset.head())

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

        for score, table_id, columns, join_map in self.__top_joinable_tables:
            self.__joinable_columns_dict[table_id] = columns

            try:
                table = self.__data_handler.get_table(table_id)
            except:
                continue
            self.__tables_dict[table_id] = table

            column_headers = [table.columns[int(col_id)] for col_id in columns.split('_')][
                             :len(self.__query_columns)]
            self.__column_headers_dict[table_id] = column_headers

    def plot_joinability_scores(self):
        scores = []
        for score, _, _, _ in self.__top_joinable_tables:
            scores += [score]

        # TODO all of top-k are accepted by default, do we want to fetch more candidates or
        # or display plot only for top-k?
        plot_data = pd.DataFrame([], columns=["Rank", "Joinability Score"])
        plot_data["Rank"] = np.arange(1, len(self.__top_joinable_tables) + 1)
        plot_data["Joinability Score"] = scores

        g = sns.catplot(data=plot_data, x="Rank", y="Joinability Score")
        g.fig.set_size_inches(10, 4)
        plt.tight_layout()
        plt.show()

    def display_joinable_table(self, rank: int) -> None:
        """

        Parameters
        ----------
        rank : int
            Rank of table in [1, k + 1].
        """
        if rank < 1 or rank > len(self.__top_joinable_tables):
            print(f"Invalid rank: {rank}. Must be in [1, k + 1]")
            return

        score, table_id, columns, _ = self.__top_joinable_tables[rank + 1]

        print(f"Score: {score} \n"
              f"Table ID: {table_id} \n"
              f"Joinable columns: {columns} \n"
              f"#rows: {self.__tables_dict[table_id].shape[0]} \n"
              f"#columns: {self.__tables_dict[table_id].shape[1]} ")

        # TODO display table
        print(self.__tables_dict[table_id].head())
        #highlight_sample = highlight_columns(tables_dict[table_id], column_headers_dict[table_id])
        #display(HTML(highlight_sample.to_html()))

    def duplicate_detection(self):
        dup = DuplicateDetection(self.__data_handler)

        #duplicate_detection_start = time.time()
        duplicate_tables = []
        for _, table_id, _, _ in self.__top_joinable_tables:
            table = self.__tables_dict[table_id]
            duplicate_tables += dup.get_duplicate_tables(table)

        self.__duplicate_relations = dup.get_relations(duplicate_tables)

        net = Network(height='1000px', width='100%', notebook=True)

        for t in self.__duplicate_relations:
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
        net.toggle_physics(True)

        # Get row values to generate html tables:
        output = ""
        for table_id in duplicate_tables:
            # print(data_handler.get_table(table_id).head(10).to_html())
            output += self.__data_handler.get_table(table_id).to_html(table_id=f"t{table_id}",
                                                                      index=False)

        # Convert CSV table to html table
        output = output + self.__input_dataset.iloc[:10, :].to_html(table_id='t0', index=False)

        with open("template.html", 'r') as file:
            filedata = file.read()

        # Replace table placeholder with actual tables html
        filedata = filedata.replace('%%tables_placeholder%%', output)

        with open("template_new.html", 'w') as file:
            file.write(filedata)

        net.prep_notebook(custom_template=True, custom_template_path="template_new.html")
        return net

        # TODO remove duplicates from top joinable tables

    def analyze_XASH_alternations(self, hash_size: int, rotation: bool, number_of_ones: int):
        def custom_xash(s: str) -> int:
            return generate_XASH(s,
                                 hash_size=hash_size,
                                 rotation=rotation,
                                 number_of_ones=number_of_ones)

        self.__data_handler.hash_function = custom_xash
        mate = MATE(self.__data_handler)

        mate.join_search(self.__input_dataset,
                         self.__query_columns,
                         10,
                         online_hash_calculation=True)
        self.__data_handler.hash_function = generate_XASH

    def correlation_calculation(self):
        cocoa = COCOA(self.__data_handler)
        self.__top_correlating_columns = cocoa.enrich_multicolumn(self.__input_dataset,
                                                                  self.__top_joinable_tables, 10,
                                                                  target_column=self.__target_column)

        # add tokenized input columns for the join
        output_dataset = self.__input_dataset.copy()
        for input_column in self.__query_columns:
            output_dataset[input_column + "_tokenized"] = self.__input_dataset[input_column].apply(
                get_cleaned_text)

        for cor, table_col_id in self.__top_correlating_columns[:3]:
            table_id = int(table_col_id.split('_')[0])
            column_id = int(table_col_id.split('_')[1])
            table = self.__tables_dict[table_id]

            # add correlation info
            new_col_name = f"{table_id}_{table.columns[column_id]}"

            self.__external_columns += [new_col_name]
            table = table.rename(columns={table.columns[column_id]: new_col_name})

            table = table.loc[:, self.__column_headers_dict[table_id] + [table.columns[column_id]]]

            output_dataset = output_dataset.merge(
                table,
                how="left",
                left_on=[col + "_tokenized" for col in self.__query_columns],
                right_on=self.__column_headers_dict[table_id],
                suffixes=('', '_extern')
            )

            # TODO fix correlation for categorical columns
            try:
                x = output_dataset[self.__target_column].astype(float)
            except ValueError:
                x = output_dataset[self.__target_column].astype('category').cat.codes
            try:
                y = output_dataset[new_col_name].astype(float)
            except ValueError:
                y = output_dataset[new_col_name].astype('category').cat.codes

            self.__pearson_dict[new_col_name] = np.corrcoef(x, y)[0]
            self.__spearman_dict[new_col_name] = cor

            # remove external join columns
            for ext_col in self.__column_headers_dict[table_id]:
                if ext_col not in self.__query_columns:
                    output_dataset = output_dataset.drop(columns=[ext_col])

            output_dataset = output_dataset[
                [c for c in output_dataset.columns if not c.endswith('_extern')]]

        output_dataset = output_dataset[
            [c for c in output_dataset.columns if not c.endswith('_tokenized')]]

        # TODO display corr coefficients for each external column
        print(self.__spearman_dict)
        print(self.__pearson_dict)

        # TODO display output dataset
        print(output_dataset)
        self.__output_dataset = output_dataset
        #output_sample = highlight_columns(output_dataset, input_columns, target=external_columns)
        #display(HTML(output_sample.to_html()))

    def plot_correlation(self):
        if self.__output_dataset is None:
            print("No output dataset available.")
            return

        # convert categorical columns to numerical ones for correlation calculation
        output_dataset = self.__output_dataset.copy()
        for col in output_dataset:
            try:
                output_dataset[col] = output_dataset[col].astype(float)
            except ValueError:
                output_dataset[col] = output_dataset[col].astype('category').cat.codes

        corr = output_dataset.corr()
        plt.figure(figsize=(10, 6))
        heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)

        plt.tight_layout()
        plt.show()

    def fit_and_evaluate_model(self):
        pass

