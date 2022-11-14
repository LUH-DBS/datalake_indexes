import pandas as pd
from data_handler import DataHandler
from collections import defaultdict
from typing import List, Tuple
from util import get_cleaned_text
import numpy as np


def fp_check(rowArray1, rowArray2):
    # Check values to check false positive
    rowvalues_t1 = rowArray1  # both are already sorted
    rowvalues_t2 = rowArray2

    ## Duplicate detection
    if len(rowvalues_t1) > len(rowvalues_t2):
        bigger_row = rowvalues_t1
        smaller_row = rowvalues_t2
    else:
        bigger_row = rowvalues_t2
        smaller_row = rowvalues_t1

    for i in range(0, len(bigger_row)):
        if i >= len(smaller_row):
            # fail
            return False
        if bigger_row[i] != smaller_row[i]:
            # fail, different values
            return False

    return True


class DuplicateDetection:
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

        self.__counter_super_key = 0
        self.__counter_fp = 0
        self.__duplicates = []
        self.__duplicate_tables = []

    def get_duplicate_tables(self, input_table: pd.DataFrame) -> List[int]:
        counter_fp = 0
        counter_superkey = 0
        dup = []
        duplicate_tables = []

        # table = input_dataset
        superKeyMapping = defaultdict(list)
        rows = defaultdict(dict)

        # generate superKeyMapping from dataframe
        for row_id, row in input_table.iloc[:1, :].iterrows():
            super_key = 0
            for _, token in row.items():
                token = get_cleaned_text(str(token))
                super_key |= self.__data_handler.hash_function(token)
                rows[0][row_id] = sorted(list(row))
                rows[1][row_id] = super_key
            superKeyMapping[super_key] += [row_id]

        in_clause = ""
        for v in rows[1].values():
            in_clause = in_clause + "'" + str(np.binary_repr(v).zfill(128)) + "',"

        in_clause = "','".join([str(np.binary_repr(v).zfill(128)) for v in rows[1].values()])

        tmp_rowid = -1
        tmp_tableid = -1
        tmp_superkey = 0
        row = []
        tableIds_length_to_load = set()

        for result in self.__data_handler.get_pl_by_super_key(in_clause):
            if (tmp_tableid != -1 and tmp_tableid != result[0]) or (
                    tmp_rowid != -1 and tmp_rowid != result[1]):
                row.sort()
                for rowId in superKeyMapping[int(tmp_superkey, 2)]:
                    if fp_check(rows[0][rowId], row):
                        dup.append((rowId, (tmp_tableid, tmp_rowid)))
                        tableIds_length_to_load.add(tmp_tableid)
                    else:
                        counter_fp = counter_fp + 1
                    counter_superkey = counter_superkey + 1
                row = []
            tmp_tableid = result[0]
            tmp_rowid = result[1]
            tmp_superkey = result[4]
            row.append(str(result[3]))

        row.sort()
        if tmp_tableid != -1:  # check that at least one row is found
            row.sort()
            for rowId in superKeyMapping[int(tmp_superkey, 2)]:
                if fp_check(rows[0][rowId], row):
                    dup.append((rowId, (tmp_tableid, tmp_rowid)))
                    tableIds_length_to_load.add(tmp_tableid)
                else:
                    counter_fp = counter_fp + 1
                counter_superkey = counter_superkey + 1

            duplicates = defaultdict(list)
            for i in dup:
                duplicates[i[1][0]].append((i[0], i[1][1]))

            if len(dup) > 0:
                # Check duplicate rows for duplicate tables
                # Get number of rows in table:
                for result in self.__data_handler.get_max_rowids(list(tableIds_length_to_load)):
                    t1_dup = []
                    t2_dup = []
                    for value in duplicates[result[0]]:
                        t1_dup.append(value[0])
                        t2_dup.append(value[1])

                    if len(set(t1_dup)) >= len(rows[0]) or len(set(t2_dup)) >= result[1] + 1:
                        if len(set(t1_dup)) >= len(t2_dup) or len(set(t2_dup)) >= len(t1_dup):
                            duplicate_tables.append(result[0])

        return duplicate_tables

    def compareTables(self, t1, t2, data) -> None:
        duplicates_local = []

        t1_data = data[1][t1]
        t2_data = data[1][t2]

        # Compare num of columns:
        if len(data[0][t1][0]) != len(data[0][t2][0]):
            return None  # Number of columns is different
        # End compare num of columns

        for row_t1 in t1_data:
            super_key_t1 = t1_data[row_t1]
            for row_t2 in t2_data:
                super_key_t2 = t2_data[row_t2]
                if len(t2_data) < 1:
                    continue

                # Compare super keys:
                if super_key_t1 == super_key_t2:
                    self.__counter_super_key += 1

                    # Check values to check false positive
                    rowvalues_t1 = list(data[0][t1][row_t1].values())
                    rowvalues_t2 = list(data[0][t2][row_t2].values())

                    rowvalues_t1.sort()
                    rowvalues_t2.sort()

                    ## Duplicate detection
                    if len(rowvalues_t1) > len(rowvalues_t2):
                        bigger_row = rowvalues_t1
                        smaller_row = rowvalues_t2
                    else:
                        bigger_row = rowvalues_t2
                        smaller_row = rowvalues_t1

                    fail = False
                    for i in range(0, len(bigger_row)):
                        if i >= len(smaller_row):
                            # fail
                            fail = True
                            break
                        if bigger_row[i] != smaller_row[i]:
                            # fail, different values
                            fail = True
                            break
                    if not fail:
                        self.__duplicates.append({"tableid_1": t1, "rowid_1": row_t1, "tableid_2": t2,
                                           "rowid_2": row_t2})
                        duplicates_local.append(
                            {"tableid_1": t1, "rowid_1": row_t1, "tableid_2": t2,
                             "rowid_2": row_t2})
                    else:
                        ## If only duplicate tables need to be found
                        # (Important: this will probably not work for subset duplicates correctly),
                        # we can completely skip this table == table comparison

                        self.__counter_fp += 1
                    ## End duplicate

        num_rows_min = min(len(t1_data), len(t2_data))
        if len(duplicates_local) >= num_rows_min and num_rows_min > 0:
            t1_dup = []
            t2_dup = []
            for value in duplicates_local:
                t1_dup.append(value['rowid_1'])
                t2_dup.append(value['rowid_2'])

            # if (len(set(t1_dup)) >= len(t1_data) or len(set(t2_dup)) >= len(t2_data)):
            #    if (len(set(t1_dup)) >= len(t2_dup) or len(set(t2_dup)) >= len(t1_dup)):
            #        duplicate_tables.append((t1,t2))
            self.__duplicate_tables.append((t1, t2))

    def get_relations(self, table_ids: List[int]) -> List[Tuple[int, int]]:
        rowValues = defaultdict(lambda: defaultdict(dict))
        rowSuperKeys = defaultdict(dict)

        for row in self.__data_handler.get_table_and_super_keys(table_ids):
            rowValues[row[0]][row[1]][row[2]] = str(row[3])
            rowSuperKeys[row[0]][row[1]] = int(row[4], 2)  # convert to int
        table_data = [rowValues, rowSuperKeys]

        self.__counter_super_key = 0
        self.__counter_fp = 0
        self.__duplicates = []
        self.__duplicate_tables = []

        for tableIds1 in table_data[0]:
            for tableIds2 in table_data[0]:
                if tableIds1 < tableIds2:
                    self.compareTables(tableIds1, tableIds2, table_data)

        return self.__duplicate_tables

