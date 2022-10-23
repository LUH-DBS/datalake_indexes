import math
import time
from operator import itemgetter
import pandas as pd
import numpy as np
from util import get_cleaned_text, create_cocoa_index
from typing import Dict, List
from data_handler import DataHandler
from tqdm import tqdm


class COCOA:
    def __init__(
            self,
            data_handler: DataHandler
    ):
        """
        Creates a COCOA instance to enrich given datasets with external data.


        Parameters
        ----------
        data_handler : DataHandler

        """
        self.__data_handler = data_handler
        self.__logger = data_handler.get_logger()

    def enrich(self,
               input_dataset: pd.DataFrame,
               k_c: int,
               k_t: int,
               query_column: str = 'query',
               target_column: str = 'target',
               join_map: np.ndarray = None) -> pd.DataFrame:
        """

        Parameters
        ----------

        Returns
        -------
        pd.DataFrame
            Input dataset, joined with top-k tables based on query column.
        """
        def generate_rank(col: pd.Series) -> pd.Series:
            """
            Returns rank of the given column.

            Parameters
            ----------
            col : Any
                Pandas dataframe column

            Returns
            -------
            Any
                Pandas dataframe column containing rank for each row.
            """
            return col.rank(na_option='bottom', method='average')

        def generate_join_map(col: pd.Series, column: Dict) -> np.ndarray:
            """
            Generates a JoinMap for a given column

            Parameters
            ----------
            col :
                Column
            column :
                Dict containing joinable column
            :return: JoinMap
            """
            vals = column.values()
            vals = [int(x) for x in vals]
            join_table = np.full(max(vals) + 1, -1)

            q = np.array(col)
            for i in np.arange(len(q)):
                x = q[i]
                index = column.get(x, -1)
                if index != -1:
                    join_table[index] = i

            return join_table

        self.__logger.info('=== Starting COCOA ===')

        # -----------------------------------------------------------------------------------------------------------
        # INPUT PREPARATION
        # -----------------------------------------------------------------------------------------------------------
        # Query preparation
        dataset = input_dataset.copy()
        dataset = dataset.apply(lambda x: x.astype(str).str.lower())
        dataset[query_column] = dataset[query_column].apply(get_cleaned_text)

        # Target preparation
        dataset['rank_target'] = generate_rank(dataset[target_column])
        target_ranks = np.array(dataset['rank_target'])
        std_target_rank = np.std(target_ranks)
        target_rank_sum = sum(target_ranks)

        # -----------------------------------------------------------------------------------------------------------
        # FINDING JOINABLE COLUMNS
        # -----------------------------------------------------------------------------------------------------------
        self.__logger.info('Finding joinable columns...')
        overlap_columns = self.__data_handler.get_joinable_columns(dataset[query_column], k_t)
        self.__logger.info('Finished.')

        if not overlap_columns:
            self.__logger.info('No joinable columns found.')
            self.__logger.info('=== Finished COCOA ===')
            return dataset     # no external content added

        # Extract table and column ids from each
        table_ids = []
        column_ids = []
        for o in overlap_columns:
            table_ids.append(int(o.split('_')[0]))
            column_ids.append(int(o.split('_')[1]))

        # -----------------------------------------------------------------------------------------------------------
        # FETCHING CONTENT OF TABLES CONTAINING JOINABLE COLUMNS
        # -----------------------------------------------------------------------------------------------------------
        self.__logger.info('Fetching content of joinable columns...')

        external_joinable_tables = self.__data_handler.get_columns(overlap_columns)
        joinable_tables_dict = {}  # store content of tables containing at least one column that is joinable with query

        for table_col_id, group in external_joinable_tables.groupby(['table_col_id']):
            keys = list(group['tokenized'])
            values = list(group['rowid'])
            item = dict(zip(keys, values))
            joinable_tables_dict[table_col_id] = item
        self.__logger.info('Finished.')

        # -----------------------------------------------------------------------------------------------------------
        # INDEX PREPARATION
        # -----------------------------------------------------------------------------------------------------------
        self.__logger.info('Fetching number of columns for each table...')
        max_column_ids = self.__data_handler.get_max_column_ids(table_ids)
        self.__logger.info('Finished.')

        # Now we compute all table_col_ids for which we need to fetch the index
        max_column_dict = max_column_ids.astype(int).set_index('tableid').to_dict()['max_col_id']
        table_col_ids = []
        for table_id in max_column_dict:
            for i in range(max_column_dict[table_id] + 1):
                table_col_ids.append(str(table_id) + '_' + str(i))

        # Datastructures in which we store the index for each table_col_id
        order_dict = {}
        binary_dict = {}
        min_dict = {}
        numerics_dict = {}

        self.__logger.info('Fetching cocoa index...')
        cocoa_index = self.__data_handler.get_cocoa_index(table_col_ids)

        for _, index in cocoa_index.iterrows():
            table_col_id = index['table_col_id']
            order_dict[table_col_id] = index['order_list']
            binary_dict[table_col_id] = index['binary_list']
            min_dict[table_col_id] = int(index['min_index'])
            numerics_dict[table_col_id] = bool(index['is_numeric'])
        self.__logger.info('Finished.')

        # -----------------------------------------------------------------------------------------------------------
        # PREPARATION
        # -----------------------------------------------------------------------------------------------------------
        input_size = len(dataset)
        column_name = []
        column_correlation = []
        join_maps = {}

        # -----------------------------------------------------------------------------------------------------------
        # CORRELATION CALCULATION
        # -----------------------------------------------------------------------------------------------------------
        self.__logger.info('Calculating correlations...')
        for i in tqdm(np.arange(len(table_ids))):
            column = column_ids[i]
            table = table_ids[i]
            max_col = max_column_dict[table]

            joinMap = generate_join_map(dataset[query_column], joinable_tables_dict[str(table) + '_' + str(column)])

            for c in np.arange(max_col + 1):
                if c == column:
                    continue

                t_c_key = f'{table}_{c}'
                join_maps[t_c_key] = joinMap

                is_numeric_column = numerics_dict[t_c_key]
                pointer = min_dict[t_c_key]
                order_index = order_dict[t_c_key]
                binary_index = binary_dict[t_c_key]

                dataset['new_external_rank'] = math.ceil(input_size / 2)
                external_rank = dataset['new_external_rank'].values

                # We use the order index to compute the ranks of each column
                if is_numeric_column:
                    # Spearman correlation coefficient
                    counter = 1
                    jump_flag = False
                    current_counter_assigned = False

                    equal_values = np.empty(len(order_index), dtype=np.int64)
                    equal_values_count = 0

                    # Average-rank for equal values:
                    while pointer != -1:
                        if jump_flag and current_counter_assigned:
                            counter += 1
                            jump_flag = False
                            current_counter_assigned = False

                        input_index = joinMap[pointer]
                        if input_index != -1:
                            external_rank[input_index] = counter
                            current_counter_assigned = True

                        # T = value[i] = value[i + 1] in column
                        if binary_index[pointer] == '1':
                            if equal_values_count:
                                equal_values[equal_values_count] = pointer
                                equal_values_count += 1

                                # We count all equal values and assign average for each
                                rank = 0
                                for j in range(0, equal_values_count):
                                    rank += external_rank[joinMap[equal_values[j]]]
                                rank = rank / equal_values_count

                                for j in range(0, equal_values_count):
                                    external_rank[joinMap[equal_values[j]]] = rank
                                equal_values_count = 0
                            jump_flag = True
                        else:
                            equal_values[equal_values_count] = pointer
                            equal_values_count += 1
                            counter += 1

                        # In the end, we have to check if the last values were equal and assign the average rank
                        if equal_values_count:
                            rank = 0
                            for j in range(0, equal_values_count):
                                rank += external_rank[joinMap[equal_values[j]]]
                            rank = rank / equal_values_count

                            for j in range(0, equal_values_count):
                                external_rank[joinMap[equal_values[j]]] = rank
                            equal_values_count = 0

                        pointer = order_index[pointer]
                    cor = np.corrcoef(dataset['rank_target'], external_rank)[0, 1]

                else:
                    # Pearson correlation coefficient
                    max_correlation = 0
                    ohe_sum = 0
                    ohe_qty = 0
                    jump_flag = False

                    while pointer != -1:
                        if jump_flag:
                            if ohe_qty > 0:
                                correlation = ((input_size * ohe_sum) - (ohe_qty * target_rank_sum)) / (
                                        std_target_rank * input_size * math.sqrt((ohe_qty * (input_size - ohe_qty))))
                                if abs(correlation) > max_correlation:
                                    max_correlation = abs(correlation)
                            ohe_qty = 0
                            ohe_sum = 0
                            jump_flag = False

                        input_index = joinMap[pointer]
                        if input_index != -1:
                            ohe_sum += target_ranks[input_index]
                            ohe_qty += 1

                        if binary_index[pointer] == 'T':
                            jump_flag = True
                        pointer = order_index[pointer]
                    cor = max_correlation
                column_name += [t_c_key]
                column_correlation += [cor]

        self.__logger.info('Finished.')

        # Now we get the topk columns with highest correlation
        overall_list = []
        for i in np.arange(len(column_correlation)):
            overall_list += [[column_correlation[i], column_name[i]]]
        sorted_list = sorted(overall_list, key=itemgetter(0), reverse=True)

        topk_table_col_ids = []
        for important_column_index in np.arange(min(k_c, len(sorted_list))):
            important_column = sorted_list[important_column_index]
            topk_table_col_ids += [important_column[1]]
        if 'new_external_rank' in dataset:
            dataset = dataset.drop('new_external_rank', axis=1)

        return topk_table_col_ids

    def enrich_multicolumn(
            self,
            input_dataset: pd.DataFrame,
            top_joinable_tables: List,
            k_c: int,
            target_column: str = 'target',
            online_index_generation: bool = False
    ) -> List:
        """

        Parameters
        ----------
        input_dataset : pd.DataFrame
            Dataset to add features to.

        top_joinable_tables : List
            Joinable tables containing Join Maps, returned by MATE.

        k_c : int
            Number of features that will be returned.

        target_column : str
            Name of the target column which is used for correlation calculation with each
            external feature.

        online_index_generation : bool
            If true, the COCOA order index is generated online.

        Returns
        -------
        pd.DataFrame
            Input dataset, joined with top-k tables based on query column.
        """
        def generate_rank(col: pd.Series) -> pd.Series:
            """
            Returns rank of the given column.

            Parameters
            ----------
            col : Any
                Pandas dataframe column

            Returns
            -------
            Any
                Pandas dataframe column containing rank for each row.
            """
            return col.rank(na_option='bottom', method='average')

        if len(top_joinable_tables) == 0:
            return []

        self.__logger.info('=== Starting COCOA multicolumn ===')
        preparation_start = time.time()

        # -----------------------------------------------------------------------------------------------------------
        # INPUT PREPARATION
        # -----------------------------------------------------------------------------------------------------------
        # Query preparation
        dataset = input_dataset.copy()

        # Target preparation
        dataset['rank_target'] = generate_rank(dataset[target_column])
        target_ranks = np.array(dataset['rank_target'])
        std_target_rank = np.std(target_ranks)
        target_rank_sum = sum(target_ranks)

        # -----------------------------------------------------------------------------------------------------------
        # EXTRACTING JOINABLE COLUMNS
        # -----------------------------------------------------------------------------------------------------------
        table_ids = [table[1] for table in top_joinable_tables]
        column_ids = [table[2].split('_') for table in top_joinable_tables]
        inverted_join_maps = [table[3] for table in top_joinable_tables]

        # -----------------------------------------------------------------------------------------------------------
        # INDEX PREPARATION
        # -----------------------------------------------------------------------------------------------------------
        self.__logger.info('Fetching number of columns for each table...')
        max_column_ids = self.__data_handler.get_max_column_ids(table_ids)
        self.__logger.info('Finished.')

        # Now we compute all table_col_ids for which we need to fetch the index
        max_column_dict = max_column_ids.astype(int).set_index('tableid').to_dict()['max_col_id']
        table_col_ids = []
        for table_id in max_column_dict:
            for i in range(max_column_dict[table_id] + 1):
                table_col_ids.append(str(table_id) + '_' + str(i))

        # Datastructures in which we store the index for each table_col_id
        order_dict = {}
        binary_dict = {}
        min_dict = {}
        numerics_dict = {}

        if not online_index_generation:
            self.__logger.info('Generating cocoa index online...')
            external_columns = self.__data_handler.get_columns(table_col_ids)
            for _, column in external_columns.iterrows():
                table_col_id = column['table_col_id']
                order_list, binary_list, min_index, is_numeric = create_cocoa_index(
                    list(column['tokenized'])
                )

                order_dict[table_col_id] = order_list
                binary_dict[table_col_id] = binary_list
                min_dict[table_col_id] = int(min_index)
                numerics_dict[table_col_id] = bool(is_numeric)
        else:
            self.__logger.info('Fetching cocoa index...')
            cocoa_index = self.__data_handler.get_cocoa_index(table_col_ids)

            for _, index in cocoa_index.iterrows():
                table_col_id = index['table_col_id']
                order_dict[table_col_id] = index['order_list']
                binary_dict[table_col_id] = index['binary_list']
                min_dict[table_col_id] = int(index['min_index'])
                numerics_dict[table_col_id] = bool(index['is_numeric'])
        self.__logger.info('Finished.')

        # -----------------------------------------------------------------------------------------------------------
        # INVERT JOIN MAPS
        # -----------------------------------------------------------------------------------------------------------
        join_maps = []

        for i in range(len(table_ids)):
            table_id = table_ids[i]
            inverted_join_map = inverted_join_maps[i]
            table_size = len(order_dict[str(table_id) + '_0'])

            join_map = np.full(table_size, -1)
            for k in range(len(inverted_join_map)):
                join_map[inverted_join_map[k]] = k
            join_maps += [join_map]

        preparation_runtime = time.time() - preparation_start

        # -----------------------------------------------------------------------------------------------------------
        # PREPARATION
        # -----------------------------------------------------------------------------------------------------------
        input_size = len(dataset)
        column_name = []
        column_correlation = []

        # -----------------------------------------------------------------------------------------------------------
        # CORRELATION CALCULATION
        # -----------------------------------------------------------------------------------------------------------
        correlation_calculation_start = time.time()
        self.__logger.info('Calculating correlations...')
        for i in tqdm(np.arange(len(table_ids))):
            columns = column_ids[i]
            table = table_ids[i]
            max_col = max_column_dict[table]

            joinMap = join_maps[i]

            for c in np.arange(max_col + 1):
                if str(c) in columns:
                    continue

                t_c_key = f'{table}_{c}'

                is_numeric_column = numerics_dict[t_c_key]
                pointer = min_dict[t_c_key]
                order_index = order_dict[t_c_key]
                binary_index = binary_dict[t_c_key]

                dataset['new_external_rank'] = math.ceil(input_size / 2)
                external_rank = dataset['new_external_rank'].values

                # We use the order index to compute the ranks of each column
                if is_numeric_column:
                    # Spearman correlation coefficient
                    counter = 1
                    jump_flag = False
                    current_counter_assigned = False

                    equal_values = np.empty(len(order_index), dtype=np.int64)
                    equal_values_count = 0

                    # Average-rank for equal values:
                    while pointer != -1:
                        if jump_flag and current_counter_assigned:
                            counter += 1
                            jump_flag = False
                            current_counter_assigned = False

                        input_index = joinMap[pointer]
                        if input_index != -1:
                            external_rank[input_index] = counter
                            current_counter_assigned = True

                        # T = value[i] = value[i + 1] in column
                        if binary_index[pointer] == '1':
                            if equal_values_count:
                                equal_values[equal_values_count] = pointer
                                equal_values_count += 1

                                # We count all equal values and assign average for each
                                rank = 0
                                for j in range(0, equal_values_count):
                                    rank += external_rank[joinMap[equal_values[j]]]
                                rank = rank / equal_values_count

                                for j in range(0, equal_values_count):
                                    external_rank[joinMap[equal_values[j]]] = rank
                                equal_values_count = 0
                            jump_flag = True
                        else:
                            equal_values[equal_values_count] = pointer
                            equal_values_count += 1
                            counter += 1

                        # In the end, we have to check if the last values were equal and assign the average rank
                        if equal_values_count:
                            rank = 0
                            for j in range(0, equal_values_count):
                                rank += external_rank[joinMap[equal_values[j]]]
                            rank = rank / equal_values_count

                            for j in range(0, equal_values_count):
                                external_rank[joinMap[equal_values[j]]] = rank
                            equal_values_count = 0

                        pointer = order_index[pointer]
                    cor = np.corrcoef(dataset['rank_target'], external_rank)[0, 1]

                else:
                    # Pearson correlation coefficient
                    max_correlation = 0
                    ohe_sum = 0
                    ohe_qty = 0
                    jump_flag = False

                    while pointer != -1:
                        if jump_flag:
                            if ohe_qty > 0:
                                correlation = ((input_size * ohe_sum) - (ohe_qty * target_rank_sum)) / (
                                        std_target_rank * input_size * math.sqrt((ohe_qty * (input_size - ohe_qty))))
                                if abs(correlation) > max_correlation:
                                    max_correlation = abs(correlation)
                            ohe_qty = 0
                            ohe_sum = 0
                            jump_flag = False

                        input_index = joinMap[pointer]
                        if input_index != -1:
                            ohe_sum += target_ranks[input_index]
                            ohe_qty += 1

                        if binary_index[pointer] == 'T':
                            jump_flag = True
                        pointer = order_index[pointer]
                    cor = max_correlation
                column_name += [t_c_key]
                column_correlation += [cor]

        self.__logger.info('Finished.')

        # Now we get the topk columns with highest correlation
        overall_list = []
        for i in np.arange(len(column_correlation)):
            overall_list += [[column_correlation[i], column_name[i]]]
        sorted_list = sorted(overall_list, key=lambda x: abs(x[0]), reverse=True)

        topk_table_col_ids = []
        for important_column_index in np.arange(min(k_c, len(sorted_list))):
            important_column = sorted_list[important_column_index]
            topk_table_col_ids += [important_column[1]]
        if 'new_external_rank' in dataset:
            dataset = dataset.drop('new_external_rank', axis=1)

        correlation_calculation_runtime = time.time() - correlation_calculation_start

        print(f"Total runtime: {preparation_runtime + correlation_calculation_runtime:.2f}s")
        print(f"Preparation runtime: {preparation_runtime:.2f}s")
        print(f"Correlation calculation runtime: {correlation_calculation_runtime:.2f}s")
        print()
        print(f"Evaluated features: {len(sorted_list)}")
        print(f"Max. correlation coefficient: {sorted_list[0][0]:.4f}")
        print(f"Min. correlation coefficient: {sorted_list[-1][0]:.4f}")

        return sorted_list[:k_c]

