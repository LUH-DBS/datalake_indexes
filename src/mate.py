import numpy as np
import pandas as pd
import time
from tqdm.notebook import tqdm_notebook
from heapq import heapify, heappush, heappop
from util import get_cleaned_text
from typing import List, Dict, Tuple
from data_handler import DataHandler
from bloom_filter import BloomFilter


class MATE:
    def __init__(
            self,
            data_handler: DataHandler,
            bf_hash_size: int = 128,
            bf_number_of_ones: int = 5,
            verbose: bool = False
    ):
        """
        Provides MATE algorithm and related functions.

        Parameters
        ----------
        data_handler : DataHandler
            Allows database communication.

        bf_hash_size : int
            Hash size for bloom filter that can be used in join_search().

        bf_number_of_ones : int
            Number of "1" bits for bloom filter that can be used in join_search().

        verbose : bool
            If true, detailed output is printed.
        """
        self.__data_handler = data_handler
        self.logging = data_handler.get_logger()

        self.__bf_hash_size = bf_hash_size
        self.__bf_number_of_ones = bf_number_of_ones

        self.__verbose = verbose

    def hash_row_values(self, row: pd.DataFrame, query_columns: List[str]) -> int:
        """
        Calculates hash for a row in a dataset.

        Parameters
        ----------
        row : pd.DataFrame
            Table row to compute hash value for.

        query_columns : List[str]
            Columns to compute hash value for.

        Returns
        -------
            Hash value for row.
        """
        hash_value = 0
        for q in query_columns:
            hash_value |= self.__data_handler.hash_function(row[q])
        return hash_value

    def hash_row_vals_bf(self, row: pd.DataFrame, query_columns: List[str]) -> str:
        """Calculates Hash value for row using Bloom Filter.

        Parameters
        ----------
        row : pd.DataFrame
            Table row to compute hash value for.

        query_columns : List[str]
            Columns to compute hash value for.

        Returns
        -------
        int
            Hash value for row.
        """
        bf = BloomFilter(6, self.__bf_hash_size, self.__bf_number_of_ones)
        for q in query_columns:
            bf.add(row[q])

        string_output = ''
        for i in bf.bit_array:
            if i:
                string_output += '1'
            else:
                string_output += '0'
        return string_output

    def evaluate_rows(self,
                      input_row: pd.Series,
                      col_dict: Dict,
                      input_data: pd.DataFrame,
                      query_columns: List[str]) -> Tuple:
        """
        Evaluate a single row.

        Parameters
        ----------


        Returns
        -------
        """
        vals = list(col_dict.values())
        query_cols_arr = np.array(query_columns)
        query_degree = len(query_cols_arr)
        matching_column_order = ''
        for q in query_cols_arr[-(query_degree - 1):]:
            q_index = list(input_data.columns.values).index(q)
            if input_row[q_index] not in vals:

                return False, ''
            else:
                for colid, val in col_dict.items():
                    if val == input_row[q_index]:
                    # if val is not None and input_row[q_index] in val:
                        matching_column_order += '_{}'.format(str(colid))
        return True, matching_column_order

    def join_search(
            self,
            input_data: pd.DataFrame,
            query_columns: List[str],
            k: int,
            k_c: int = 500,
            min_join_ratio: int = 0,
            use_hash_optimization: bool = True,
            use_bloom_filter: bool = False,
            online_hash_calculation: bool = False,
            stats: dict = None
    ) -> List:
        """
        Finds top-k joinable tables based on selected query columns.

        Parameters
        ----------
        input_data : pd.DataFrame
            Input dataset containing query columns.

        query_columns : List[str]
            List of query columns based on which MATE discovers joinable tables.

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

        stats : dict
            Dictionary to store algorithm stats in.

        Returns
        -------
        List
            Top joinable tables.
        """

        # -----------------------------------------------------------------------------------------------------------
        # INPUT PREPARATION
        # -----------------------------------------------------------------------------------------------------------
        if self.__verbose:
            print("Preparing input dataset...")
        orig_input_data = input_data.copy()
        orig_input_data = orig_input_data.applymap(lambda x: get_cleaned_text(x)).replace('', np.nan).replace('nan', np.nan)\
            .replace('unknown', np.nan)
        input_data = input_data.drop_duplicates(subset=query_columns)

        input_data = input_data.applymap(lambda x: get_cleaned_text(x)).replace('', np.nan).replace('nan', np.nan)\
            .replace('unknown', np.nan)

        for q in query_columns:
            input_data.dropna(subset=[q], inplace=True)
        input_size = len(input_data)

        if len(input_data) == 0:
            return []

        is_linear = (self.__data_handler.hash_function is None)

        row_block_size = 1
        total_runtime = 0
        db_runtime = 0
        total_match = 0
        total_fp = 0
        total_pruned = 0
        max_SQL_parameter_size = 1000000
        total_approved = 0
        total_filtered = 0
        hash_verification_time = 0
        evaluation_time = 0
        relevant_rows_time = 0
        table_dictionary_generation_runtime = 0

        if use_bloom_filter:
            input_data['SuperKey'] = input_data.apply(
                lambda row: self.hash_row_vals_bf(row, query_columns), axis=1)
        elif not is_linear:
            input_data.loc[:, 'SuperKey'] = input_data.apply(
                lambda row: self.hash_row_values(row, query_columns), axis=1)

        # NEW for join map: index has to be a separate row
        input_data.loc[:, 'MateRowID'] = np.arange(input_size)

        g = input_data.groupby([query_columns[0]])

        gd = {}
        for key, item in g:
            gd[key] = np.array(g.get_group(key))

        if not is_linear:
            super_key_index = list(input_data.columns.values).index('SuperKey')

        # NEW for join map:
        row_id_index = list(input_data.columns.values).index('MateRowID')
        index_to_mate_row_id = input_data['MateRowID'].to_dict()

        if online_hash_calculation:
            token_dict_for_hash = {}

        if self.__verbose:
            print("Done.")

        # -----------------------------------------------------------------------------------------
        # FETCHING JOINABLE TABLES
        # -----------------------------------------------------------------------------------------
        if self.__verbose:
            print("Fetching joinable tables based on first query column...")

        top_joinable_tables = []  # each item includes: Tableid, joinable_rows
        heapify(top_joinable_tables)

        if not is_linear and not online_hash_calculation:
            table_row = self.__data_handler.get_concatinated_posting_list_with_hash(
                input_data[query_columns[0]])
        elif online_hash_calculation:
            table_row = self.__data_handler.get_concatinated_posting_list(
                input_data[query_columns[0]])

        table_dictionary_generation_start_runtime = time.time()
        table_dictionary = {}
        for i in table_row:
            if str(i) == 'None':
                continue
            tableid = int(i.split(';')[0].split('_')[0])
            if tableid in table_dictionary:
                table_dictionary[tableid] += [i]
            else:
                table_dictionary[tableid] = [i]
        table_dictionary_generation_runtime += time.time() -\
                                               table_dictionary_generation_start_runtime
        input_row_ids = []
        candidate_external_row_ids = []
        candidate_external_col_ids = []
        candidate_input_rows = []
        candidate_table_rows = []
        candidate_table_ids = []

        overlaps_dict = {}
        pruned = False

        join_maps = {}

        if self.__verbose:
            print("Done.")

        # -----------------------------------------------------------------------------------------
        # PRUNING
        # -----------------------------------------------------------------------------------------
        if self.__verbose:
            print("Running hash-based row filtering...")
        iterator = sorted(table_dictionary,
                          key=lambda k: len(table_dictionary[k]), reverse=True)[:k_c]
        if self.__verbose:
            iterator = tqdm_notebook(iterator, position=0, leave=True)

        for tableid in iterator:
            set_of_rowids = set()
            hitting_posting_list_concatinated = table_dictionary[tableid]
            if len(top_joinable_tables) >= k and top_joinable_tables[0][0] >= len(
                    hitting_posting_list_concatinated):
                pruned = True
            if len(hitting_posting_list_concatinated) < min_join_ratio:
                pruned = True

            if pruned:
                total_pruned += 1

            if online_hash_calculation:
                hit_rows = pd.Series([x.split(';')[0] for x in hitting_posting_list_concatinated])
                if len(hit_rows) > max_SQL_parameter_size:
                    hit_row_values = self.__data_handler.get_pl_by_table_and_rows_incremental(hit_rows, max_SQL_parameter_size)  # table_row, col, tokenized
                else:
                    hit_row_values = self.__data_handler.get_pl_by_table_and_rows(hit_rows)  # table_row, col, tokenized
                for i in hit_row_values:
                    if i[0] not in token_dict_for_hash:
                        token_dict_for_hash[i[0]] = {}
                    token_dict_for_hash[i[0]][i[1]] = i[2]

            already_checked_hits = 0
            for hit in sorted(hitting_posting_list_concatinated):
                if len(top_joinable_tables) >= k and (
                        (len(hitting_posting_list_concatinated) - already_checked_hits +
                         len(set_of_rowids)) < top_joinable_tables[0][0]):
                    break
                tablerowid = hit.split(';')[0]
                rowid = tablerowid.split('_')[1]
                colid = hit.split(';')[1].split('$')[0].split('_')[0]
                token = hit.split(';')[1].split('$')[0].split('_')[1]
                if not is_linear and not online_hash_calculation:
                    superkey = int(hit.split('$')[1], 2)

                # SuperKey Generation for this row
                if online_hash_calculation:
                    if tablerowid not in token_dict_for_hash:
                        continue
                    col_dictionary = token_dict_for_hash[tablerowid]
                    superkey = 0
                    for col_key in sorted(col_dictionary):
                        h = self.__data_handler.hash_function(str(col_dictionary[col_key]))
                        superkey = superkey | h

                hash_verification_start_time = time.time()
                relevant_rows_start_time = time.time()
                start_time = time.time()

                if token not in gd:
                    continue
                relevant_input_rows = gd[token]

                relevant_rows_time += (time.time() - relevant_rows_start_time)
                for input_row in relevant_input_rows:
                    if not use_hash_optimization or is_linear or ((input_row[super_key_index] | superkey) == superkey):
                        # NEW for join map, store input rowid separately
                        input_row_ids += [int(input_row[row_id_index])]

                        candidate_external_row_ids += [rowid]
                        set_of_rowids.add(rowid)
                        candidate_external_col_ids += [colid]
                        # candidate_input_rows += [input_row]

                        # NEW for join map
                        candidate_input_rows += [input_row[:row_id_index]]

                        candidate_table_ids += [tableid]
                        candidate_table_rows += ['{}_{}'.format(tableid, rowid)]
                    else:
                        total_filtered += 1

                total_runtime += (time.time() - start_time)
                hash_verification_time += (time.time() - hash_verification_start_time)
                already_checked_hits += 1

            evaluation_start_time = time.time()
            start_time = time.time()
            if pruned or len(candidate_external_row_ids) >= row_block_size:
                if len(candidate_external_row_ids) == 0:
                    break
                candidate_input_rows = np.array(candidate_input_rows)
                candidate_table_ids = np.array(candidate_table_ids)
                temp_start_db_time = time.time()
                if len(candidate_table_rows) > max_SQL_parameter_size:
                    pls = self.__data_handler.get_pl_by_table_and_rows_incremental(
                        candidate_table_rows, max_SQL_parameter_size)
                else:
                    pls = self.__data_handler.get_pl_by_table_and_rows(candidate_table_rows)

                db_runtime += (time.time() - temp_start_db_time)

                # contains rowid that each rowid has dict that maps colids to tokenized
                table_row_dict = {}

                for i in pls:
                    if i[0] not in table_row_dict:
                        table_row_dict[str(i[0])] = {}
                        table_row_dict[str(i[0])][str(i[1])] = str(i[2])
                    else:
                        table_row_dict[str(i[0])][str(i[1])] = str(i[2])
                for i in np.arange(len(candidate_table_rows)):
                    if candidate_table_rows[i] not in table_row_dict:
                        continue
                    col_dict = table_row_dict[candidate_table_rows[i]]
                    match, matched_columns = self.evaluate_rows(candidate_input_rows[i],
                                                                col_dict,
                                                                input_data,
                                                                query_columns)

                    total_approved += 1
                    if match:
                        total_match += 1
                        complete_matched_columns = f'{str(candidate_external_col_ids[i])}' \
                                                   f'{matched_columns}'
                        if candidate_table_ids[i] not in overlaps_dict:
                            overlaps_dict[candidate_table_ids[i]] = {}

                        if complete_matched_columns in overlaps_dict[candidate_table_ids[i]]:
                            overlaps_dict[candidate_table_ids[i]][complete_matched_columns] += 1
                        else:
                            overlaps_dict[candidate_table_ids[i]][complete_matched_columns] = 1

                        # NEW FOR JOIN MAPS
                        if candidate_table_ids[i] not in join_maps:
                            join_maps[candidate_table_ids[i]] = {}

                        if complete_matched_columns not in join_maps[candidate_table_ids[i]]:
                            join_maps[candidate_table_ids[i]][complete_matched_columns] = np.full(
                                [input_size], -1)

                        # make sure that smallest index of each group is assigned
                        if join_maps[candidate_table_ids[i]][complete_matched_columns][
                            input_row_ids[i]] == -1 \
                                or join_maps[candidate_table_ids[i]][complete_matched_columns][
                            input_row_ids[i]] > int(candidate_external_row_ids[i]):
                            join_maps[candidate_table_ids[i]][complete_matched_columns][
                                input_row_ids[i]] = candidate_external_row_ids[i]
                    else:
                        total_fp += 1

                for tbl in set(candidate_table_ids):
                    if tbl in overlaps_dict and len(overlaps_dict[tbl]) > 0:
                        join_keys = max(overlaps_dict[tbl], key=overlaps_dict[tbl].get)
                        joinability_score = overlaps_dict[tbl][join_keys]
                        if k <= len(top_joinable_tables):
                            if top_joinable_tables[0][0] < joinability_score:
                                popped_table = heappop(top_joinable_tables)
                                heappush(top_joinable_tables, [joinability_score, tbl, join_keys])
                        else:
                            heappush(top_joinable_tables, [joinability_score, tbl, join_keys])

                input_row_ids = []      # NEW for join map
                candidate_external_row_ids = []
                candidate_external_col_ids = []
                candidate_input_rows = []
                candidate_table_rows = []
                candidate_table_ids = []

                overlaps_dict = {}
            total_runtime += (time.time() - start_time)
            evaluation_time += (time.time() - evaluation_start_time)
            if pruned:
                break

        if self.__verbose:
            print("Done.")

        # -----------------------------------------------------------------------------------------------------------
        # CREATE FINAL JOIN MAPS
        # -----------------------------------------------------------------------------------------------------------
        if self.__verbose:
            print("Generating join maps...")

        # TODO make more efficient
        for table_id in join_maps:
            for columns in join_maps[table_id]:
                final_join_map = np.full(len(orig_input_data), -1)

                for _, group in orig_input_data.groupby(query_columns):
                    group_index = index_to_mate_row_id[group.iloc[0, :].name]

                    for row_index, _ in group.iterrows():
                        final_join_map[row_index] = join_maps[table_id][columns][group_index]

                join_maps[table_id][columns] = final_join_map

        top_joinable_tables_with_join_maps = [[score - 1, table_id, columns, join_maps[table_id][columns]] for score, table_id, columns in top_joinable_tables]

        if stats is not None:
            stats["table_dict_runtime"] = table_dictionary_generation_runtime
            stats["mate_runtime"] = total_runtime
            stats["db_runtime"] = db_runtime
            stats["total_filtered"] = total_filtered
            stats["total_approved"] = total_approved
            stats["matching_rows"] = total_match
            stats["total_fp"] = total_fp
            stats["precision"] = total_match / max(total_approved, 1)

        if self.__verbose:
            print("Done.")

        return sorted(top_joinable_tables_with_join_maps, key=lambda x: x[0], reverse=True)
