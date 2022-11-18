import time
from typing import Any, List, Tuple, Callable, Union
from util import get_cleaned_text, create_cocoa_index, generate_XASH
import pandas as pd
import numpy as np
from io import StringIO
import os
from tqdm import tqdm
import logging
import csv
import arff
from collections import defaultdict

#import vertica_python

# Vertica
'''
MAIN_COLUMNS = [{'name': 'tokenized', 'type': 'VARCHAR(255)'},
                {'name': 'tableid', 'type': 'INT NOT NULL'},
                {'name': 'colid', 'type': 'INT NOT NULL'},
                {'name': 'rowid', 'type': 'INT NOT NULL'},
                {'name': 'table_col_id', 'type': 'VARCHAR(14) NOT NULL'}]

COCOA_COLUMNS = [{'name': 'is_numeric', 'type': 'BOOLEAN'},
                 {'name': 'min_index', 'type': 'INT'},
                 {'name': 'order_list', 'type': 'LONG VARCHAR(17446248)'},
                 {'name': 'binary_list', 'type': 'LONG VARCHAR(4361562)'}]
MATE_COLUMNS = [{'name': 'super_key', 'type': 'BINARY(16)'}]
'''

# Postgres
MAX_TOKEN_LENGTH = 200
MAIN_COLUMNS = [{'name': 'tokenized', 'type': f'VARCHAR({MAX_TOKEN_LENGTH})'},
                {'name': 'tableid', 'type': 'INT NOT NULL'},
                {'name': 'colid', 'type': 'INT NOT NULL'},
                {'name': 'rowid', 'type': 'INT NOT NULL'},
                {'name': 'table_col_id', 'type': 'TEXT NOT NULL'}]

COLUMNS_COLUMNS = [{'name': 'tableid', 'type': 'INT NOT NULL'},
                   {'name': 'colid', 'type': 'INT NOT NULL'},
                   {'name': 'header', 'type': 'TEXT'},
                   {'name': 'header_tokenized', 'type': 'TEXT'}]

TABLE_INFO_COLUMNS = [{'name': 'tableid', 'type': 'INT NOT NULL'},
                      {'name': 'dataset_name', 'type': 'TEXT'},
                      {'name': 'max_row_id', 'type': 'INT NOT NULL'},
                      {'name': 'max_col_id', 'type': 'INT NOT NULL'}]

COCOA_COLUMNS = [{'name': 'table_col_id', 'type': 'TEXT NOT NULL'},
                 {'name': 'is_numeric', 'type': 'BOOLEAN'},
                 {'name': 'min_index', 'type': 'INT'},
                 {'name': 'order_list', 'type': 'INT[]'},
                 {'name': 'binary_list', 'type': 'BIT VARYING'}]
MATE_COLUMNS = [{'name': 'super_key', 'type': 'BIT(128)'}]


class DataHandler:
    """

    Parameters
    ----------
    conn : Any

    main_table : str

    max_col_table : str

    max_row_table : str

    use_helper_tables : str

    cocoa : bool

    mate : bool

    hash_function : str

    """
    def __init__(
            self,
            conn: Any,
            main_table: str,
            column_headers_table: str,
            table_info_table: str,
            cocoa_index_table: str,
            cocoa: bool = True,
            mate: bool = True,
            logger: Any = logging,
            hash_function: Callable[[str], int] = generate_XASH
    ):

        if not cocoa and not mate:
            raise Exception('Please choose at least one tool: [COCOA, MATE].')

        if mate:
            self.__mate_hash_dict = {}

        self.__conn = conn
        self.__cur = conn.cursor()

        self.main_table = main_table
        self.cocoa_index_table = cocoa_index_table
        self.column_headers_table = column_headers_table
        self.table_info_table = table_info_table

        self.__cocoa = cocoa
        self.__mate = mate

        self.hash_function: Callable[[str], int] = hash_function

        self.__logger = logger

        self.__cur_id = 1         # next table id that will be assigned
        self.__tables = []        # stores all tables and their ids

        self.__index_updated = False   # handler can only be used if index is up-to-date
        self.__db_ready = False        # data can only be inserted after the db preparation is done

        self.__inserted_tables = 0
        self.__file_errors = 0
        self.__data_errors = 0

    # -----------------------------------------------------------------------------------------------------------
    # PRIVATE
    # -----------------------------------------------------------------------------------------------------------
    def __str__(self):
        result = 'Data Handler\n'

        result += '\nDatabase Settings\n'
        result += '----------------------------------------------------\n'
        result += f'Main table: {self.main_table}\n'
        result += f'COCOA index table: {self.cocoa_index_table}\n'
        result += f'Column headers table: {self.column_headers_table}\n'

        result += '\nTool Settings\n'
        result += '----------------------------------------------------\n'
        result += f'COCOA enabled: {self.__cocoa}\n'
        result += f'MATE enabled: {self.__mate}\n'

        result += '\nCurrent State\n'
        result += '----------------------------------------------------\n'
        result += f'Read tables: {len(self.__tables)}\n'
        result += f'Database ready: {self.__db_ready}\n'
        result += f'Index up-to-date: {self.__index_updated}\n'

        return result

    def __commit(self) -> None:
        """
        Commits the current database transaction.
        """
        self.__cur.execute('COMMIT;')

    def __prepare_db(self) -> None:
        """
        Creates required tables in the database.
        """
        # main table
        self.__cur.execute(f'CREATE TABLE IF NOT EXISTS {self.main_table} (' +
                           ', '.join([f'{column["name"]} {column["type"]}' for column in MAIN_COLUMNS]) +
                           ');')

        # column headers
        self.__cur.execute(f'CREATE TABLE IF NOT EXISTS {self.column_headers_table} (' +
                           ', '.join([f'{column["name"]} {column["type"]}' for column in COLUMNS_COLUMNS]) +
                           ');')

        # table info
        self.__cur.execute(f'CREATE TABLE IF NOT EXISTS {self.table_info_table} (' +
                           ', '.join([f'{column["name"]} {column["type"]}' for column in TABLE_INFO_COLUMNS]) +
                           ');')

        self.__commit()

        # Create additional columns and tables
        if self.__cocoa:
            self.__cur.execute(f'CREATE TABLE IF NOT EXISTS {self.cocoa_index_table} (' +
                               ', '.join([f'{column["name"]} {column["type"]}' for column in COCOA_COLUMNS]) +
                               ');')

        if self.__mate:
            for column in MATE_COLUMNS:
                try:
                    self.__cur.execute(f'ALTER TABLE {self.main_table} ADD COLUMN {column["name"]} {column["type"]}')
                except Exception as e:
                    if 'exists' not in str(e):
                        self.__logger.error(e)

        self.__commit()

    def __init(self) -> None:
        """
        Initializes required database tables and datastructures.
        """
        self.__prepare_db()

        # If there is already data inside the table we have to set the table id counter to max(table id) + 1
        self.__cur.execute(f'SELECT COUNT(*) FROM {self.main_table};')
        if int(self.__cur.fetchall()[0][0]) > 0:
            self.__cur.execute(f'SELECT MAX(tableid) FROM {self.main_table};')
            self.__cur_id = int(self.__cur.fetchall()[0][0]) + 1

        self.__db_ready = True

    def __create_db_indexes(self) -> None:
        """
        Creates indexes for all tables in the database.
        """
        self.__logger.info('Creating indexes. This might take some time...')
        self.__cur.execute(f'CREATE INDEX IF NOT EXISTS {self.main_table}_tokenized_idx '
                           f'ON {self.main_table} (tokenized);')
        self.__cur.execute(f'CREATE INDEX IF NOT EXISTS {self.main_table}_tableid_rowid_idx '
                           f' ON {self.main_table} (tableid, rowid);')
        self.__cur.execute(f'CREATE INDEX IF NOT EXISTS {self.main_table}_table_col_id_idx '
                           f' ON {self.main_table} (table_col_id);')
        self.__cur.execute(f'CREATE INDEX IF NOT EXISTS {self.main_table}_tableid_idx '
                           f' ON {self.main_table} (tableid);')

        self.__cur.execute(f'CREATE INDEX IF NOT EXISTS {self.column_headers_table}_table_col_id_colid_idx '
                           f' ON {self.column_headers_table} (tableid, colid);')

        self.__cur.execute(f'CREATE INDEX IF NOT EXISTS {self.table_info_table}_tableid_idx '
                           f' ON {self.table_info_table} (tableid);')

        if self.__cocoa:
            self.__cur.execute(f'CREATE INDEX IF NOT EXISTS {self.cocoa_index_table}_tableid_colid_idx '
                               f' ON {self.cocoa_index_table} (table_col_id);')

        self.__commit()
        self.__logger.info('Indexes ready.')

    def __read_json(self, filepath: str) -> Tuple[str, pd.DataFrame]:
        """


        Parameters
        ----------
        filepath : str
        """
        try:
            table = pd.read_json(filepath)
        except Exception as _:
            self.__file_errors += 1
            raise ValueError()
        return filepath.split('/')[-1], table

    def __add_table(self, filepath: str) -> None:
        """
        Insert a table into table list before creating the index.

        Parameters
        ----------
        filepath : str
           Filepath
        """
        self.__tables.append(filepath)

    def __index_table(self, table_id: int, table: pd.DataFrame, name: str) -> None:
        # -----------------------------------------------------------------------------------------------------------
        # COLUMN HEADERS
        # -----------------------------------------------------------------------------------------------------------
        headers_buffer = StringIO()  # all headers are inserted in the end

        for col_id in range(len(table.columns)):
            header = str(table.columns[col_id])
            headers_buffer.write('\t'.join([str(table_id),
                                            str(col_id),
                                            header,
                                            get_cleaned_text(header)]) + '\n')

        # -----------------------------------------------------------------------------------------------------------
        # CELL VALUES
        # -----------------------------------------------------------------------------------------------------------
        table_buffer = StringIO()
        table.reset_index()
        table = table.copy()    # avoid fragmentation
        max_row_id = 0
        max_col_id = 0
        for row_id, row in table.iterrows():
            super_key = 0
            if self.__mate:
                for _, token in row.items():
                    super_key |= self.hash_function(token)
            max_row_id = max(max_row_id, int(row_id))

            col_id = 0
            for _, token in row.items():
                max_col_id = max(max_col_id, col_id)

                table_col_id = str(table_id) + '_' + str(col_id)
                if token is None:
                    token = '\\N'

                value_list = [token, str(table_id), str(col_id), str(row_id), table_col_id]

                if self.__mate:
                    bin_super_key = bin(super_key)[2:]

                    # TODO replace with dynamic hash size
                    bin_super_key = bin_super_key.zfill(128)

                    value_list += [bin_super_key]

                table_buffer.write('\t'.join(value_list) + '\n')

                col_id += 1

        # -----------------------------------------------------------------------------------------------------------
        # COCOA INDEX
        # -----------------------------------------------------------------------------------------------------------
        if self.__cocoa:
            for col_id in range(len(table.columns)):
                table_col_id = str(table_id) + "_" + str(col_id)

                min_index, order_list, binary_list, is_numeric = create_cocoa_index(table.iloc[:, col_id])

                joint_order_list = ','.join([str(val) for val in order_list])
                joint_binary_list = ''.join(binary_list)

                try:
                    self.__cur.execute(f'INSERT INTO {self.cocoa_index_table} '
                                       f'VALUES ('
                                       f'   \'{table_col_id}\','
                                       f'   {is_numeric},'
                                       f'   {min_index}, '
                                       f'   ARRAY[{joint_order_list}], '
                                       f'   \'{joint_binary_list}\''
                                       f');')
                except Exception as e:
                    logging.error(e)
                    exit()
                    print(f'Error at table_col_id {table_col_id}.')
                    print(e)
                    continue

        # -----------------------------------------------------------------------------------------------------------
        # INSERTION
        # -----------------------------------------------------------------------------------------------------------
        table_buffer.seek(0)
        columns = ['tokenized', 'tableid', 'colid', 'rowid', 'table_col_id']
        if self.__mate:
            columns += ['super_key']
        self.__cur.copy_from(table_buffer,
                             self.main_table,
                             sep='\t',
                             null='\\N',
                             columns=columns)
        print(f"Inserted buffer of length {len(table_buffer.getvalue())}")
        logging.info("INSERTED CELLS!!!!")

        # Insert column headers
        headers_buffer.seek(0)
        self.__cur.copy_from(headers_buffer,
                             self.column_headers_table,
                             sep='\t',
                             null='\\N',
                             columns=[column['name'] for column in COLUMNS_COLUMNS])

        self.__cur.execute(f'INSERT INTO {self.table_info_table} '
                           f'VALUES ({table_id}, \'{name}\', {max_row_id}, {max_col_id});')

        self.__cur.commit()
        # self.__commit()

    def __create_inverted_index(self) -> None:
        """
        Generates the inverted index for all tables in table list and stores it in the database.
        """
        if self.__mate:
            self.__logger.info(f'Creating inverted index and MATE index for {len(self.__tables)} tables.')
        else:
            self.__logger.info(f'Creating inverted index for {len(self.__tables)} tables.')

        # -----------------------------------------------------------------------------------------------------------
        # READ FILES
        # -----------------------------------------------------------------------------------------------------------
        for filepath in tqdm(self.__tables, ascii=True):
            ending = filepath.split('.')[-1]
            if ending == 'csv':
                read_func = self.read_csv
            elif ending == 'json':
                read_func = self.__read_json
            elif ending == 'parquet':
                read_func = self.read_parquet
            elif ending == 'arff':
                read_func = self.read_arff
            else:
                logging.info('Invalid file format: ' + filepath.split('.')[-1])
                self.__file_errors += 1
                continue

            table: pd.DataFrame()
            table_name = ''
            try:
                result = read_func(filepath)
                if result:
                    table_name = result[0]
                    table = result[1]
            except Exception as e:
                logging.info('Unable to read file: ' + filepath)
                logging.error(e)
                continue

            # --------------------------------------------------------------------------------------------------------
            # TABLE CLEANUP AND INDEXING
            # --------------------------------------------------------------------------------------------------------
            table = table.applymap(str).applymap(get_cleaned_text).applymap(lambda x: x[:MAX_TOKEN_LENGTH])
            if table.shape[0] == 0:
                self.__data_errors += 1
                continue

            try:
                self.__index_table(self.__cur_id, table, table_name)
                self.__cur_id += 1
                self.__inserted_tables += 1
            except Exception as _:
                self.__data_errors += 1

        self.__logger.info(f'Inserted {self.__inserted_tables} tables.')
        self.__logger.info(f'Encountered {self.__file_errors} file errors.')
        self.__logger.info(f'Encountered {self.__data_errors} data format errors.')

    '''
    def __create_cocoa_index(self) -> None:
        """
        Generates and inserts the COCOA index and stores it in the database.
        """
        logging.info('Creating COCOA index...')
        self.__cur.execute(f'SELECT MAX(tableid) '
                           f'FROM {self.table_info_table};')
        num_tables = self.__cur.fetchall()[0][0]
        if num_tables is None:
            logging.info(f'No tables to create index for.')
            return

        logging.info(f'Starting index creation for {num_tables} tables...')

        for table_id in tqdm(range(1, num_tables + 1), ascii=True):
            try:
                self.__cur.execute(f'SELECT table_col_id, ARRAY_AGG(rowid), ARRAY_AGG(tokenized) '
                                   f'FROM {self.main_table} '
                                   f'WHERE tableid = {table_id} '
                                   f'GROUP BY table_col_id;')
            except Exception as e:
                print(f'Error at table {table_id}.')
                print(e)
                continue

            for column in self.__cur.fetchall():
                table_col_id = column[0]
                row_ids = column[1]
                tokens = column[2]
                tokens_by_rowid_asc = [str(val[1]) for val in sorted(zip(row_ids, tokens), key=lambda x: x[0])]

                min_index, final_order_list, final_binary_list, is_numeric = create_cocoa_index(tokens_by_rowid_asc)

                joint_order_list = ','.join([str(val) for val in final_order_list])
                joint_binary_list = ''.join(final_binary_list)

                try:
                    self.__cur.execute(f'INSERT INTO {self.cocoa_index_table} '
                                       f'VALUES ('
                                       f'   \'{table_col_id}\','
                                       f'   {is_numeric},'
                                       f'   {min_index}, '
                                       f'   ARRAY[{joint_order_list}], '
                                       f'   \'{joint_binary_list}\''
                                       f');')
                except Exception as e:
                    print(f'Error at table_col_id {table_col_id}.')
                    print(e)
                    continue
            self.__commit()
    '''

    # -----------------------------------------------------------------------------------------------------------
    # PUBLIC, PREPARATION
    # -----------------------------------------------------------------------------------------------------------
    def get_logger(self) -> Any:
        """

        Returns
        -------
        Any
            This data handler's logger.
        """
        return self.__logger

    def clean_up_db(self) -> None:
        """
        Deletes all tables in the database, which were created earlier.

        Can be called before init() to delete existing data.
        """
        self.__cur.execute(f'DROP TABLE IF EXISTS {self.main_table};')
        self.__cur.execute(f'DROP TABLE IF EXISTS {self.column_headers_table};')
        self.__cur.execute(f'DROP TABLE IF EXISTS {self.table_info_table};')
        self.__cur.execute(f'DROP TABLE IF EXISTS {self.cocoa_index_table};')

        self.__commit()

    def add_tables_folder(self, path: str) -> None:
        """


        Parameters
        ----------
        path : str
        """
        read_tables = 0
        self.__logger.info(f'Adding tables from: {path}')
        for filename in tqdm(os.listdir(path), ascii=True):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                self.__add_table(filepath)
                read_tables += 1
        self.__logger.info(f'Found {read_tables} tables.')

    def read_arff(self, filepath) -> Tuple[str, pd.DataFrame]:
        """
        Reads a dataset in arff format from file.

        Parameters
        ----------
        filepath : str
            Dataset path.

        Returns
        -------
        Tuple[str, pd.DataFrame]
            Dataset name and content.
        """
        try:
            data = defaultdict(list)
            arff_gen = arff.load(filepath)
            for row in arff_gen:
                n_values = len(row._values)
                keys = [key for key in list(row._data.keys())[:n_values]]

                for key in keys:
                    data[key] += [row._data[key]]

            table = pd.DataFrame(data)
        except:
            self.__file_errors += 1
            raise ValueError()

        return filepath.split('/')[-1], table

    def read_csv(self, filepath) -> Tuple[str, pd.DataFrame]:
        """
        Reads a dataset in csv format from file.

        Parameters
        ----------
        filepath : str
            Dataset path.

        Returns
        -------
        Tuple[str, pd.DataFrame]
            Dataset name and content.
        """
        def extract_delimiter_from_line(file: str) -> str:
            """
            :param file:
            :return: csv delimiter
            """
            sniffer = csv.Sniffer()
            try:
                with open(file, 'r') as f:
                    line = f.readline()
            except UnicodeDecodeError:
                with open(file, 'r', encoding='latin-1') as f:
                    line = f.readline()

            try:
                dialect = sniffer.sniff(line)
                return str(dialect.delimiter)
            except csv.Error:
                raise

        def read_dataframe_from_file(file: str, delim: str) -> pd.DataFrame:
            """
            :param file:
            :param delim:
            :return: dataframe
            """
            try:
                df = pd.read_csv(file, delimiter=delim, dtype=str)
            except UnicodeDecodeError:
                df = pd.read_csv(file, delimiter=delim, encoding='latin-1', dtype=str)
            except pd.errors.ParserError:
                raise
            return df

        try:
            delimiter = extract_delimiter_from_line(filepath)
        except csv.Error as _:
            self.__file_errors += 1
            raise ValueError()
        try:
            table = read_dataframe_from_file(filepath, delimiter)
        except pd.errors.ParserError as _:
            self.__file_errors += 1
            raise ValueError()
        return filepath.split('/')[-1], table

    def read_parquet(self, filepath: str) -> Tuple[str, pd.DataFrame]:
        """


        Parameters
        ----------
        filepath : str

        Returns
        -------
        Tuple[str, pd.DataFrame]
            Filename and dataframe.
        """
        return filepath.split('/')[-1], pd.read_parquet(filepath)

    def update_index(self) -> None:
        """
        Updates the inverted index as well as the indexes for COCOA and MATE in the database.
        """
        if not self.__db_ready:
            self.__init()

        self.__create_inverted_index()
        self.__create_db_indexes()

        self.__index_updated = True

    # -----------------------------------------------------------------------------------------------------------
    # PUBLIC, SHARED FUNCTIONS
    # -----------------------------------------------------------------------------------------------------------
    def get_table(self, table_id: int) -> pd.DataFrame:
        """


        Parameters
        ----------
        table_id : int

        Returns
        -------
        pd.DataFrame

        """
        self.__cur.execute(f'SELECT tokenized, colid, rowid '
                           f'FROM {self.main_table} '
                           f'WHERE tableid = {table_id} '
                           f'ORDER BY colid, rowid;')
        content = pd.DataFrame(self.__cur.fetchall(), columns=['tokenized', 'colid', 'rowid'])

        table = pd.DataFrame()
        for col_id, column_content in content.groupby(['colid']):
            table[col_id] = list(column_content['tokenized'])

        self.__cur.execute(f'SELECT header '
                           f'FROM {self.column_headers_table} '
                           f'WHERE tableid = {table_id} '
                           f'ORDER BY colid;')
        table.columns = [header[0] for header in self.__cur.fetchall()]

        return table

    # -----------------------------------------------------------------------------------------------------------
    # PUBLIC, COCOA FUNCTIONS
    # -----------------------------------------------------------------------------------------------------------
    def get_joinable_columns(self, tokens: pd.Series, k_t: int) -> List[str]:
        """


        Parameters
        ----------
        tokens : pd.Series

        k_t : int

        Returns
        -------
        List[str]

        """
        distinct_clean_values = tokens.unique()
        joint_distinct_values = '\',\''.join(distinct_clean_values)

        # TODO check if this  WHERE ct >= 3 correct
        self.__cur.execute(f'SELECT ol.table_col_id '
                           f'FROM ('
                           f'    SELECT table_col_id, COUNT(DISTINCT tokenized) as ct '
                           f'    FROM {self.main_table} '
                           f'    WHERE tokenized IN (\'{joint_distinct_values}\') '
                           f'    GROUP BY table_col_id '
                           f') as ol '
                           f'WHERE ct > 0'
                           f'ORDER BY ct DESC '
                           f'LIMIT {k_t};')

        return [row[0] for row in self.__cur.fetchall()]

    def get_columns(self, table_col_ids: List[str]) -> pd.DataFrame:
        """


        Parameters
        ----------
        table_col_ids : List[str]

        Returns
        -------
        pd.DataFrame

        """
        joint_table_ids = '\',\''.join(table_col_ids)
        start = time.time()
        self.__cur.execute(f'SELECT table_col_id, tokenized, rowid '
                           f'FROM {self.main_table} '
                           f'WHERE table_col_id IN (\'{joint_table_ids}\') '
                           f'ORDER BY table_col_id, rowid;')

        return pd.DataFrame(
            self.__cur.fetchall(),
            columns=['table_col_id', 'tokenized', 'rowid'])

    def get_cocoa_index(self, table_col_ids: List[str]) -> pd.DataFrame:
        """


        Parameters
        ----------
        table_col_ids : List[str]

        Returns
        -------
        pd.DataFrame

        """
        joint_table_ids = '\',\''.join(table_col_ids)
        self.__cur.execute(f'SELECT table_col_id, is_numeric, min_index, order_list, binary_list '
                           f'FROM {self.cocoa_index_table} '
                           f'WHERE table_col_id IN (\'{joint_table_ids}\');')
        return pd.DataFrame(
            self.__cur.fetchall(),
            columns=['table_col_id', 'is_numeric', 'min_index', 'order_list', 'binary_list'])

    def get_max_column_ids(self, table_ids: List[int]) -> pd.DataFrame:
        """


        Parameters
        ----------
        table_ids : List[int]

        Returns
        -------
        pd.DataFrame

        """
        joint_table_ids = '\',\''.join([str(i) for i in table_ids])

        self.__cur.execute(f'SELECT tableid, max_col_id '
                           f'FROM {self.table_info_table} '
                           f'WHERE tableid IN (\'{joint_table_ids}\');')

        return pd.DataFrame(self.__cur.fetchall(), columns=['tableid', 'max_col_id'])

    # -----------------------------------------------------------------------------------------------------------
    # PUBLIC, MATE FUNCTIONS
    # -----------------------------------------------------------------------------------------------------------
    def get_concatinated_posting_list(self, token_list: pd.Series) -> List[str]:
        """


        Parameters
        ----------
        token_list : pd.Series

        Returns
        -------
        List[str]

        """
        distinct_clean_values = token_list.unique()
        joint_distinct_values = '\',\''.join(distinct_clean_values)

        self.__cur.execute(f'SELECT concat(concat(concat(concat(concat(concat('
                           f'  tableid,\'_\'), rowid), \';\'), colid), \'_\'), tokenized) '
                           f'FROM {self.main_table} '
                           f'WHERE tokenized IN (\'{joint_distinct_values}\');')

        return [item for sublist in self.__cur.fetchall() for item in sublist]

    def get_concatinated_posting_list_with_hash(self, token_list: pd.Series) -> List[str]:
        """


        Parameters
        ----------
        token_list : pd.Series

        Returns
        -------
        List[str]

        """
        distinct_clean_values = token_list.unique()
        joint_distinct_values = '\',\''.join(distinct_clean_values)
        hash_column = MATE_COLUMNS[0]['name']

        self.__cur.execute(f'SELECT concat(concat(concat(concat(concat(concat(concat(concat('
                           f'  tableid,\'_\'), rowid), \';\'), colid), \'_\'), tokenized), \'$\'), {hash_column}) '
                           f'FROM {self.main_table} '
                           f'WHERE tokenized IN (\'{joint_distinct_values}\');')

        return [item for sublist in self.__cur.fetchall() for item in sublist]

    def get_pl_by_table_and_rows_incremental(
            self,
            token_list: Union[List[str], pd.Series],
            max_parameter_length: int
    ) -> List[str]:
        """


        Parameters
        ----------
        token_list : Union[List[str], pd.Series]

        max_parameter_length : int

        Returns
        -------
        List[str]

        """
        pl = []
        for i in range(0, len(token_list), max_parameter_length):
            sublist = token_list[i:i + max_parameter_length]
            distinct_clean_values = list(set(sublist))
            joint_distinct_values = '\',\''.join(distinct_clean_values)
            tables = '\',\''.join(list(set([x.split('_')[0] for x in sublist])))
            rows = '\',\''.join(list(set([x.split('_')[1] for x in sublist])))

            if len(list(set([x.split('_')[0] for x in token_list]))) == 1:
                table = list(set([x.split('_')[0] for x in token_list]))[0]
                query = f'SELECT concat(concat(tableid, \'_\'), rowid), colid, tokenized ' \
                        f'FROM {self.main_table} ' \
                        f'WHERE tableid = {table} ' \
                        f'AND rowid IN(\'{rows}\') ' \
                        f'AND concat(concat(tableid, \'_\'), rowid) IN (\'{joint_distinct_values}\');'
            else:
                query = f'SELECT concat(concat(tableid, \'_\'), rowid), colid, tokenized ' \
                        f'FROM {self.main_table} ' \
                        f'WHERE tableid IN (\'{tables}\') ' \
                        f'AND rowid IN(\'{rows}\') ' \
                        f'AND concat(concat(tableid, \'_\'), rowid) IN (\'{joint_distinct_values}\');'

            self.__cur.execute(query)
            pl += self.__cur.fetchall()

        return pl

    def get_pl_by_table_and_rows(self, joint_list: Union[List[str], pd.Series]) -> List[str]:
        """


        Parameters
        ----------
        joint_list : pd.Series

        Returns
        -------
        List[str]

        """
        distinct_clean_values = list(set(joint_list))
        joint_distinct_values = '\',\''.join(distinct_clean_values)
        tables = '\',\''.join(list(set([x.split('_')[0] for x in joint_list])))
        rows = '\',\''.join(list(set([x.split('_')[1] for x in joint_list])))

        if len(list(set([x.split('_')[0] for x in joint_list]))) == 1:
            table = list(set([x.split('_')[0] for x in joint_list]))[0]
            query = f'SELECT concat(concat(tableid, \'_\'), rowid), colid, tokenized ' \
                    f'FROM {self.main_table} ' \
                    f'WHERE tableid = {table} ' \
                    f'AND rowid IN(\'{rows}\');'
        else:
            query = f'SELECT concat(concat(tableid, \'_\'), rowid), colid, tokenized ' \
                    f'FROM {self.main_table} ' \
                    f'WHERE tableid IN (\'{tables}\') ' \
                    f'AND rowid IN(\'{rows}\') ' \
                    f'AND concat(concat(tableid, \'_\'), rowid) IN (\'{joint_distinct_values}\');'

        self.__cur.execute(query)
        return self.__cur.fetchall()

    def get_pl_by_super_key(self, in_clause: str) -> Any:
        self.__cur.execute(f'SELECT tableid, rowid, colid, tokenized, super_key '
                           f'FROM {self.main_table} '
                           f'WHERE super_key IN (\'{in_clause}\') '
                           f'ORDER BY tableid, rowid, colid')
        return self.__cur.fetchall()

    def get_max_rowids(self, table_ids: List[int]) -> Any:
        joint_table_ids = ','.join([str(val) for val in table_ids])
        self.__cur.execute(f'SELECT tableid, MAX(rowid) '
                           f'FROM {self.main_table} '
                           f'WHERE tableid IN ({joint_table_ids}) '
                           f'GROUP BY tableid')
        return self.__cur.fetchall()

    def get_table_and_super_keys(self, table_ids: List[int]):
        joint_table_ids = ','.join([str(val) for val in table_ids])
        self.__cur.execute(
            f'SELECT tableid, rowid, colid, tokenized, super_key '
            f'FROM {self.main_table} '
            f'WHERE tableid '
            f'IN ({joint_table_ids})')
        return self.__cur.fetchall()
