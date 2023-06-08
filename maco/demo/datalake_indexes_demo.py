import os
import contextlib
import numpy as np
import pandas as pd
from maco.data_handler import DataHandler
from maco.cocoa import COCOA
from maco.mate import MATE
from maco.duplicate_detection import DuplicateDetection
from maco.util import get_cleaned_text, generate_XASH
from maco.machine_learning import fit_model
import psycopg2
from typing import List, Tuple, Union
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from collections import defaultdict
import numpy.ma as ma
import itertools
from tqdm.notebook import tqdm_notebook

import warnings
warnings.filterwarnings('ignore')

# seaborn colors
COLOR_ORANGE = 'rgb(250, 180, 130)'
COLOR_GREEN = 'rgb(141, 229, 161)'
COLOR_PURPLE = 'rgb(208, 188, 255)'

DB_CONFIG = {
    "host": "herkules.dbs.uni-hannover.de",
    "dbname": "pdb",
    "user": "macodemo",
    "password": "demonstration"
}


QUERY_COLUMNS = {
    "who": ["Country"],
    "movie": ["Director Name", "Movie Title"]
}

TARGET_COLUMN = {
    "who": "Life expectancy",
    "movie": "IMDB Score"
}


def highlight_cells(
        table: pd.DataFrame,
        query_columns: List[str],
        row_ids: np.ndarray = None,
        target_column: str = None,
        ext_columns: List[str] = None
):
    def highlight_query(s):
        return f'background-color: {COLOR_GREEN}; color: #000'

    def highlight_target(s):
        return f'background-color: {COLOR_ORANGE}; color: #000'

    def highlight_external(s):
        return f'background-color: {COLOR_PURPLE}; color: #000'

    if row_ids is None:
        row_ids = np.arange(len(table))

    # keep only row ids that are actually in table
    row_ids = row_ids[np.isin(row_ids, table.index)]

    table = table.style.applymap(highlight_query,
                                 subset=pd.IndexSlice[row_ids, query_columns])
    if target_column:
        table = table.applymap(highlight_target, subset=pd.IndexSlice[:, target_column])

    if ext_columns:
        table = table.applymap(highlight_external, subset=pd.IndexSlice[:, ext_columns])

    return table


class DatalakeIndexesDemo:
    """
    Provides all required tools to run the datalake indexes demonstration scenario.
    """
    def __init__(
            self,
            datalake: str,
            display_table_rows: int = 5
    ):

        conn = psycopg2.connect(**DB_CONFIG)

        if datalake == "gittables":
            datalake = "gittables_demo"
            self.__data_handler = DataHandler(
                conn,
                main_table=f"{datalake}_main_tokenized",
                column_headers_table=f"{datalake}_column_headers",
                table_info_table=f"{datalake}_table_info",
                cocoa_index_table=f"{datalake}_cocoa_index"
            )
        elif datalake == "open_data":
            datalake = "cafe"
            self.__data_handler = DataHandler(
                conn,
                main_table=f"{datalake}_main_tokenized",
                column_headers_table=f"{datalake}_column_tokenized",
                table_info_table=f"{datalake}_table_info",
                cocoa_index_table=f"{datalake}_cocoa_index"
            )
        elif datalake == "webtable":
            self.__data_handler = DataHandler(
                conn,
                main_table=f"mate_main_tokenized",
                column_headers_table="",
                table_info_table="",
                cocoa_index_table="order_index"
            )
        else:
            raise ValueError("Invalid datalake:", datalake)

        self.__input_dataset = None
        self.__query = None
        self.__target = None
        self.__output_dataset = None
        self.__external_columns = []

        self.__tables_dict: dict = {}     # stores tables, key is tableId
        self.__joinable_columns_dict: dict = {}
        self.__column_headers_dict: dict = {}
        self.__top_joinable_tables: List[Tuple[int, int, List[int], np.ndarray]] = []

        self.__top_correlating_columns: List = [Tuple[float, str]]
        self.__spearman_dict = {}
        self.__pearson_dict = {}

        self.__display_table_rows = display_table_rows

        sns.set(font_scale=1.3)
        sns.color_palette("pastel")

        devnull = open(os.devnull, 'w')
        contextlib.redirect_stderr(devnull)

    def load_dataset(
            self,
            dataset: str,
            query: Union[str, List[str]] = None,
            target: str = None
    ) -> None:
        # if not provided fall back to default
        if query is None:
            query = QUERY_COLUMNS[dataset]
        if isinstance(query, str):
            query = [query]

        if target is None:
            target = TARGET_COLUMN[dataset]

        sep = ","
        if dataset == "who":
            sep = ";"
        df = pd.read_csv(f"./datasets/{dataset}.csv", sep=sep)

        # cleanup
        df.columns = [str(col).strip() for col in df.columns]
        df = df.fillna(df.mean())

        for query_column in query:
            if query_column not in df.columns:
                print(f"{query_column} not in input dataset.")
                return
        self.__query = query

        if target not in df.columns:
            print(f"{target} not in input dataset.")
            print("Available columns:")
            print(list(df.columns))
            return
        self.__target = target

        # for WHO: group over years
        if dataset == "who":
            df = df.groupby(query).mean(numeric_only=True).reset_index()

        self.__input_dataset = df

        html_table = highlight_cells(self.__input_dataset.head(self.__display_table_rows),
                                     query_columns=self.__query,
                                     target_column=self.__target).to_html()
        display(HTML(html_table))

    def joinability_discovery(
            self,
            k: int = 10,
            k_c: int = 500,
            min_join_ratio: int = 0,
            use_hash_optimization: bool = True,
            use_bloom_filter: bool = False,
            online_hash_calculation: bool = False,
            verbose: bool = True
    ) -> None:
        """
        Finds joinable tables within the datalake using MATE.

        Parameters
        ----------
        k : int
            Top-k joinable tables are returned.

        k_c : int
            Number of candidate tables to evaluate based on first query column.

        min_join_ratio : int
            Minimum number of joinable rows a table must contain.

        use_hash_optimization : bool
            If false, it runs join search without hash-based filtering.

        use_bloom_filter : bool
            If true, bloom filter is used for hashing the input cells.

        online_hash_calculation : bool
            If true, row hashes are calculated during filtering and not fetched from the database.

        verbose : bool
            If true, detailed output is printed.
        """
        stats = {}

        mate = MATE(self.__data_handler, verbose=verbose)
        self.__top_joinable_tables = mate.join_search(
            self.__input_dataset,
            self.__query,
            k,
            k_c=k_c,
            min_join_ratio=min_join_ratio,
            use_hash_optimization=use_hash_optimization,
            use_bloom_filter=use_bloom_filter,
            online_hash_calculation=online_hash_calculation,
            stats=stats
        )

        if verbose:
            print("Fetching tables...")
        for score, table_id, columns, join_map in self.__top_joinable_tables:
            self.__joinable_columns_dict[table_id] = columns

            try:
                table = self.__data_handler.get_table(table_id)
            except Exception as e:
                print(f"Could not fetch table {table_id}")
                continue
            self.__tables_dict[table_id] = table

            column_headers = [table.columns[int(col_id)] for col_id in columns.split('_')][
                             :len(self.__query)]
            self.__column_headers_dict[table_id] = column_headers

        if verbose:
            print("Done.")

        # -----------------------------------------------------------------------------------------------------------
        # STATISTICS
        # -----------------------------------------------------------------------------------------------------------
        print("--------------------------------------------")
        print("Runtime:")
        print("--------------------------------------------")
        print(f"Fetching candidate tables: {stats['table_dict_runtime']:.2f}s")
        print(f"MATE filtering:            {stats['mate_runtime']:.2f}s")
        print(f"Fetching row values:       {stats['db_runtime']:.2f}s")
        print()
        print("--------------------------------------------")
        print("Statistics:")
        print("--------------------------------------------")
        print(f"Hash-based filtered rows:  {stats['total_filtered']}")
        print(f"Hash-based approved rows:  {stats['total_approved']}")
        print(f"Matching rows:             {stats['matching_rows']}")
        print(f"FP rows:                   {stats['total_fp']}")
        print(f"Precision:                 {stats['precision']:.3f}")

    def plot_joinability_scores(self):
        scores = []
        for score, _, _, _ in self.__top_joinable_tables:
            scores += [score]

        plot_data = pd.DataFrame([], columns=["Table Rank", "Joinability Score"])
        plot_data["Table Rank"] = np.arange(1, len(self.__top_joinable_tables) + 1)
        plot_data["Joinability Score"] = scores

        g = sns.catplot(data=plot_data, x="Table Rank", y="Joinability Score")
        g.fig.set_size_inches(8, 3)
        plt.tight_layout()
        plt.show()

    def display_joinable_table(self, rank: int) -> None:
        """

        Parameters
        ----------
        rank : int
            Rank of table.
        """
        if rank < 1 or rank > len(self.__top_joinable_tables):
            print(f"Invalid rank: {rank}. Must be in [1, {len(self.__top_joinable_tables)}]")
            return

        score, table_id, columns, join_map = self.__top_joinable_tables[rank - 1]

        print(f"Joinability score: {score} \n"
              f"Table ID: {table_id} \n"
              f"#rows: {self.__tables_dict[table_id].shape[0]} \n"
              f"#columns: {self.__tables_dict[table_id].shape[1]} ")

        # extract matching row ids from join map
        highlighted_rows = np.where(join_map >= 0)[0]
        if len(highlighted_rows) == 0:
            highlighted_rows = None
        try:
            table = self.__tables_dict[table_id]
            #table.to_csv("../temp_data/joinable_table.csv")

            html_table = highlight_cells(table.head(),
                                         self.__column_headers_dict[table_id],
                                         row_ids=highlighted_rows).to_html()
            display(HTML(html_table))
        except Exception as e:
            print(e)
            print(self.__tables_dict[table_id].head())

    def keep_joinable_tables(self, k: int):
        self.__top_joinable_tables = self.__top_joinable_tables[:k]

    def duplicate_detection(self):
        dup = DuplicateDetection(self.__data_handler)

        duplicate_tables = []
        for _, table_id, _, _ in self.__top_joinable_tables:
            table = self.__tables_dict[table_id]
            duplicate_tables += dup.get_duplicate_tables(table)

        if len(duplicate_tables) == 0:
            print("No duplicate tables found.")

        self.__duplicate_relations = dup.get_relations(duplicate_tables)

        net = Network(height='1000px', width='100%', notebook=True)

        for t in self.__duplicate_relations:
            net.add_node(t[0], str(t[0]))
            net.add_node(t[1], str(t[1]))
            net.add_edge(t[0], t[1], physics=False)

        # net.add_node(0,"0")
        # for t in duplicate_tables_first:
        #    net.add_node(t,str(t))
        #    net.add_edge(0, t)

        net.show_buttons(filter_=['physics'])
        net.set_edge_smooth("dynamic")
        net.toggle_stabilization(True)
        net.toggle_physics(True)

        # Get row values to generate html tables:
        output = ""
        for table_id in duplicate_tables:
            table = self.__data_handler.get_table(table_id)
            #table = table.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt')]}]).set_properties(**{'font-size': '12pt'})
            output += table.to_html(table_id=f"t{table_id}", index=False)

        # Convert CSV table to html table
        output = output + self.__input_dataset.head().to_html(table_id='t0', index=False)

        template_path = "./maco/demo/template.html"
        new_template_path = "./maco/demo/template_new.html"

        print("wd:", os.getcwd())

        # Google Colab
        if not os.path.exists(template_path):
            template_path = "./template.html"
            new_template_path = "./template_new.html"

        with open(template_path, 'r') as file:
            filedata = file.read()

        # Replace table placeholder with actual tables html
        filedata = filedata.replace('%%tables_placeholder%%', output)

        with open(new_template_path, 'w') as file:
            file.write(filedata)

        net.prep_notebook(custom_template=True, custom_template_path="./maco/demo/template_new.html")
        return net

    def remove_duplicates(self):
        # group relations by first table in tuple
        duplicates_dict = defaultdict(list)
        for t1, t2 in self.__duplicate_relations:
            duplicates_dict[t1] += [t2]

        # merge dictionary into groups
        remove_tables = []  # tables that will be removed
        for t1 in duplicates_dict:
            for t2 in duplicates_dict[t1]:
                if t2 in duplicates_dict:
                    duplicates_dict[t1] += duplicates_dict[t2]
                    duplicates_dict[t2] = []
            duplicates_dict[t1] = list(set(duplicates_dict[t1]))
            remove_tables += duplicates_dict[t1]

        top_joinable_tables_filtered = []
        for i in range(len(self.__top_joinable_tables)):
            if self.__top_joinable_tables[i][1] not in remove_tables:
                top_joinable_tables_filtered += [self.__top_joinable_tables[i]]

        n_unfiltered = len(self.__top_joinable_tables)
        self.__top_joinable_tables = top_joinable_tables_filtered
        n_filtered = len(self.__top_joinable_tables)
        print(f"Reduced the number of joinable tables from {n_unfiltered} to {n_filtered}.")

    def analyze_XASH_alternations(self):
        alternations = [
            [64, 128, 256, 512],    # hash size
            [True, False],          # rotation
            [5]                     # number of ones
        ]

        iterations = np.product([len(alt) for alt in alternations])

        results = defaultdict(list)
        for hash_size, rotation, number_of_ones in tqdm_notebook(itertools.product(*alternations),
                                                        total=iterations):
            def custom_xash(s: str) -> int:
                return generate_XASH(s,
                                     hash_size=hash_size,
                                     rotation=rotation,
                                     number_of_ones=number_of_ones)
            stats = {}

            self.__data_handler.hash_function = custom_xash
            mate = MATE(self.__data_handler)

            mate.join_search(self.__input_dataset,
                             self.__query,
                             10,
                             online_hash_calculation=True,
                             stats=stats)

            results["Hash size"] += [hash_size]
            results["Rotation"] += ["Enabled" if rotation else "Disabled"]
            results["Precision"] += [stats["precision"]]

        result_data = pd.DataFrame(results).sort_values(by=["Precision"])

        plt.figure(figsize=(8, 6))
        g = sns.barplot(data=result_data, x="Hash size", y="Precision", hue="Rotation")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

        self.__data_handler.hash_function = generate_XASH

    def correlation_calculation(
            self,
            k_c: int = 10,
            online_index_generation: bool = False
    ):
        """
        k_c : int
            Number of features that will be returned.

        online_index_generation : bool
            If true, the COCOA order index is generated online.
        """
        stats = {}
        cocoa = COCOA(self.__data_handler)

        self.__top_correlating_columns = cocoa.enrich_multicolumn(
            self.__input_dataset,
            self.__top_joinable_tables,
            k_c=k_c,
            online_index_generation=online_index_generation,
            target_column=self.__target,
            stats=stats
        )
        print("--------------------------------------------")
        print("Runtime:")
        print("--------------------------------------------")
        print(f"Total runtime: {stats['total_runtime']:.2f}s")
        print(f"Preparation runtime: {stats['preparation_runtime']:.2f}s")
        print(f"Correlation calculation runtime: {stats['correlation_calculation_runtime']:.2f}s")
        print()
        print("--------------------------------------------")
        print("Statistics:")
        print("--------------------------------------------")
        print(f"Evaluated features: {stats['evaluated_features']}")
        print(f"Max. correlation coefficient: {stats['max_corr_coeff']:.4f}")

    def plot_correlation_coefficients(self):
        corr_coeffs = []
        for corr, table_col_id, is_numeric in self.__top_correlating_columns:
            corr_coeffs += [abs(corr)]

        plot_data = pd.DataFrame([], columns=["Rank", "Joinability Score"])
        plot_data["Rank"] = np.arange(1, len(self.__top_correlating_columns) + 1)
        plot_data["|Correlation Coefficient|"] = corr_coeffs

        g = sns.catplot(data=plot_data, x="Rank", y="|Correlation Coefficient|")
        g.fig.set_size_inches(8, 3)
        plt.tight_layout()
        plt.show()

    def add_external_features(self, ranks: List[int]):
        """

        Parameters
        ----------
        ranks : List[int]
            Rank of last feature that is added.
        """
        # add tokenized input columns for the join
        output_dataset = self.__input_dataset.copy()
        self.__external_columns = []
        for input_column in self.__query:
            output_dataset[input_column + "_tokenized"] = self.__input_dataset[input_column].apply(
                get_cleaned_text)

        for rank in ranks:
            cor, table_col_id, is_numeric = self.__top_correlating_columns[rank - 1]
            table_id = int(table_col_id.split('_')[0])
            column_id = int(table_col_id.split('_')[1])
            table = self.__tables_dict[table_id]

            # add correlation info
            new_col_name = str(table_col_id) + ":" + str(table.columns[column_id])

            self.__external_columns += [new_col_name]
            table = table.rename(columns={table.columns[column_id]: new_col_name})

            table = table.loc[:, self.__column_headers_dict[table_id] + [table.columns[column_id]]]

            # only use the first matching token for the join
            table = table.drop_duplicates(subset=self.__column_headers_dict[table_id])

            output_dataset = output_dataset.merge(
                table,
                how="left",
                left_on=[col + "_tokenized" for col in self.__query],
                right_on=self.__column_headers_dict[table_id],
                suffixes=('', '_extern')
            )

            if is_numeric:
                output_dataset[new_col_name] = output_dataset[new_col_name].astype(float)

                self.__pearson_dict[new_col_name] = abs(
                    ma.corrcoef(ma.masked_invalid(output_dataset[self.__target].astype(float)),
                                ma.masked_invalid(output_dataset[new_col_name].astype(float)))[0][1]
                )
                self.__spearman_dict[new_col_name] = abs(cor)

            # remove external join columns
            for ext_col in self.__column_headers_dict[table_id]:
                if ext_col not in self.__query:
                    output_dataset = output_dataset.drop(columns=[ext_col])

            output_dataset = output_dataset[
                [c for c in output_dataset.columns if not c.endswith('_extern')]]

        output_dataset = output_dataset[
            [c for c in output_dataset.columns if not c.endswith('_tokenized')]]

        output_dataset = output_dataset.fillna(output_dataset.mean())
        self.__output_dataset = output_dataset
        print(output_dataset.shape)
        #output_dataset.to_csv("../temp_data/output_dataset.csv")

        output_sample = highlight_cells(
            output_dataset.head(),
            self.__query,
            target_column=self.__target,
            ext_columns=self.__external_columns
        )
        display(HTML(output_sample.to_html()))

    def plot_spearman_pearson(self):
        corr_data = defaultdict(list)
        for ext_col in self.__spearman_dict:
            for corr_type, corr_dict in zip(["Spearman", "Pearson"],
                                            [self.__spearman_dict, self.__pearson_dict]):
                corr_data["External feature"] += [ext_col]
                corr_data["Correlation coefficient"] += [corr_dict[ext_col]]
                corr_data["Type"] += [corr_type]

        plt.figure(figsize=(10, 4))
        g = sns.barplot(data=pd.DataFrame(corr_data), x="External feature", y="Correlation coefficient",
                        hue="Type")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
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

        # replace correlation coefficients for external features by COCOA results
        for ext_col in self.__external_columns:
            if ext_col in self.__spearman_dict:
                corr.loc[self.__target, ext_col] = self.__spearman_dict[ext_col]
                corr.loc[ext_col, self.__target] = self.__spearman_dict[ext_col]
        #corr.to_csv("../temp_data/correlation.csv")

        plt.figure(figsize=(14, 6))
        heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)

        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.show()

    def fit_and_evaluate_model(self, only_input=False, verbosity: int = 0):
        errors = []

        if only_input:
            datasets = zip(["Input"], [self.__input_dataset.copy()])
        else:
            datasets = zip(["Input", "Enriched"],
                           [self.__input_dataset.copy(), self.__output_dataset.copy()])

        for dataset_name, dataset in datasets:
            print(f"{dataset_name}: Learning ML model...")
            features = []
            for col in dataset:
                if col not in self.__query:
                    features += [col]

            mse, pfis = fit_model(self.__output_dataset[features], features, self.__target, verbosity=verbosity)
            print(f"Done.")

            errors += [mse]

            # plot feature importance

            plt.figure(figsize=(20, 20))
            plt.xticks(rotation=90)
            sns.barplot(data=pfis, x="feature", y="importance", errorbar="sd")

            plt.tight_layout()
            plt.show()

        if only_input:
            dataset_names = ["Input"]
        else:
            dataset_names = ["Input", "Enriched"]

        mse_data = pd.DataFrame({
            "Dataset": dataset_names,
            "RMSE": errors
        })

        plt.figure(figsize=(5, 4))
        if len(errors) > 1:
            rel_error = (1 - errors[1] / errors[0]) * 100
            plt.title(f"RMSE reduced by {rel_error:.2f}%")

        sns.barplot(data=mse_data, x="Dataset", y="RMSE")

        plt.tight_layout()
        plt.show()



    def get_table(self, table_id: int):
        return self.__data_handler.get_table(table_id)







