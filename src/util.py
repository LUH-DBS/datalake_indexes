import re
import numpy as np
import pandas as pd
from collections import Counter
import math
from typing import Dict, List, Tuple
import hashlib


def get_cleaned_text(text):
    # if text is None or len(str(text)) == 1:
    #     return ''
    stopwords = ['a', 'the', 'of', 'on', 'in', 'an', 'and', 'is', 'at', 'are', 'as', 'be', 'but', 'by', 'for', 'it',
                 'no', 'not', 'or', 'such', 'that', 'their', 'there', 'these', 'to', 'was', 'with', 'they', 'will',
                 'v', 've', 'd']

    cleaned = re.sub('[\W_]+', ' ', str(text).encode('ascii', 'ignore').decode('ascii')).lower()
    feature_one = re.sub(' +', ' ', cleaned).strip()
    punct = [',', '.', '!', ';', ':', '?', "'", '"', '\t', '\n']

    for x in stopwords:
        feature_one = feature_one.replace(' {} '.format(x), ' ')
        if feature_one.startswith('{} '.format(x)):
            feature_one = feature_one[len('{} '.format(x)):]
        if feature_one.endswith(' {}'.format(x)):
            feature_one = feature_one[:-len(' {}'.format(x))]

    for x in punct:
        feature_one = feature_one.replace('{}'.format(x), ' ')
    return feature_one


def create_cocoa_index(values: List[str]) -> Tuple[int, List[int], List[str], bool]:
    """
    Creates order index for given column/list of values as presented in the COCOA paper.

    Parameters
    ----------
    values : List[int]
        List of column values to generate COCOA index for.

    Returns
    -------
    Tuple[int, List[int], List[str], bool]
        min_index : int
            Index of minimum element in order_list.

        order_list : List[int]
            List of indexes based on which the ranks can be constructed in linear time.

        binary_list : List[str]
            List of boolean values, True if current value is equal to next value.

        is_num : bool
            True, if column consists of only numeric values.
    """
    def is_numeric(s: str) -> bool:
        """
        Checks if given value is numeric.

        Parameters
        ----------
        s : str
            Input value.

        Returns
        -------
        bool
            True, if value is numeric.
        """
        if s.lower() == 'nan':
            return True
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_numeric_list(value_list: List[str]) -> bool:
        """
        Checks if given list is numeric.

        Parameters
        ----------
        value_list : List[str]
            Input values.

        Returns
        -------
        bool
            True, if all input values are numeric.
        """
        for k in np.arange(len(value_list)):
            if value_list[k] is None or value_list[k] == '':
                value_list[k] = np.nan
            else:
                value_list[k] = value_list[k]

        result = [val for val in value_list if is_numeric(str(val))]
        return len(result) == len(value_list)

    rows = np.arange(0, len(values), 1)
    is_num = is_numeric_list(values)
    if is_num:
        for i in np.arange(len(values)):
            if values[i] is None or values[i] == '':
                values[i] = np.nan
        values = [float(i) for i in values]
    else:
        for i in np.arange(len(values)):
            if values[i] is None:
                values[i] = ''
        values = [str(i) for i in values]
    ranks = list(pd.Series(values).rank(na_option='bottom', method='average'))

    rows_sorted_based_on_ranks = [x for _, x in sorted(zip(ranks, rows))]
    min_index = rows_sorted_based_on_ranks[0]  # starting point in the order index
    order_list = np.empty(len(rows), dtype=int)
    binary_list = np.empty(len(rows), dtype=str)
    sorted_ranks = np.sort(ranks).copy()
    for i in np.arange(len(rows) - 1):
        order_list[i] = rows_sorted_based_on_ranks[i + 1]
        # if both values are NaN we treat them as equal
        if np.isnan(sorted_ranks[i]) and np.isnan(sorted_ranks[i + 1]) \
                or sorted_ranks[i] == sorted_ranks[i + 1]:
            binary_list[i] = '0'
        else:
            binary_list[i] = '1'
    order_list[len(rows) - 1] = -1  # Maximum value
    binary_list[len(rows) - 1] = '0'  # Maximum value

    final_order_list = [x for _, x in
                        sorted(zip(rows_sorted_based_on_ranks, order_list))]
    final_binary_list = [x for _, x in
                         sorted(zip(rows_sorted_based_on_ranks, binary_list))]

    return min_index, final_order_list, final_binary_list, is_num


def XASH(
        token: str,
        hash_size: int = 128,
        rotation: bool = True
) -> int:
    """
    Computes XASH value of given token.

    Parameters
    ----------
    token : str
        Token that is hashed.

    hash_size : int
        Size of the hash value, i.e. number of bits.

    rotation : bool
        If true, XASH rotation is used.

    Returns
    -------
        XASH value.
    """
    number_of_ones = 5

    if token in ['', 'None', ' ', '\'\'']:
        return 0

    char = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    segment_size_dict = {64: 1, 128: 3, 256: 6, 512: 13}
    segment_size = segment_size_dict[hash_size]
    length_bit_start = 37 * segment_size
    result = 0
    cnt_dict = Counter(token)
    selected_chars = [y[0] for y in
                      sorted(cnt_dict.items(), key=lambda i: (i[1], i[0]), reverse=False)[:number_of_ones]]

    for c in selected_chars:
        if c not in char:
            continue
        indices = [i for i, ltr in enumerate(token) if ltr == c]
        mean_index = np.mean(indices)
        token_size = len(token)
        for i in np.arange(segment_size):
            if mean_index <= ((i + 1) * token_size / segment_size):
                location = char.index(c) * segment_size + i
                break
        result = result | int(math.pow(2, location))

    n = int(result)

    if rotation:
        # Normalize the rotation based on the location of length bit.
        # For instance in 128 bits that length segment has 17 bits and hash segment has 111 bits,
        # if the token has length of 10, the normalized number of rotation bits is 111 * 10 / 17
        d = int((length_bit_start * (len(token) % (hash_size - length_bit_start))) / (
                hash_size - length_bit_start))

        int_bits = int(length_bit_start)
        x = n << d
        y = n >> (int_bits - d)
        r = int(math.pow(2, int_bits))
        result = int((x | y) % r)

    result = int(result) | int(
        math.pow(2, len(token) % (hash_size - length_bit_start)) * math.pow(2, length_bit_start))

    return result


def BF(token: str) -> int:
    """
    Computes value of given token.

    :param token: Token
    :return: XASH of token
    """

    # TODO: implement bloom filter

    raise NotImplementedError


def MD5(token: str) -> int:
    """
    Computes MD5 hash value of given token.

    Parameters
    ----------
    token : str
        Token that is hashed.

    Returns
    -------
        MD5 hash value.
    """

    return int(hashlib.md5(token).hexdigest(), 16)

