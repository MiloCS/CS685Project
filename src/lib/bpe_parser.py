import numpy as np
import os
import json
from multiprocessing import Pool
from itertools import repeat


def helper_process_row(row):
        # row string " 123 456 789\n" -> integer list [123, 456, 789]
        row = row.strip().split()
        row = list(map(int, row))
        return row

def read_bpe_data(data_path, n_pools=8):
    """
    Reads in the BPE data from the given path into an (N, M) integer numpy array.
    Samples are padded with -1
    N = Number of samples
    M = Maximum number of tokens in an example
    """
    with open(data_path, "r") as f:
        lines = f.readlines()    

    with Pool(n_pools) as p:
    # list_data = list(map(process_row, lines))
        list_data = list(p.map(helper_process_row, lines))

    # Find the length of the longest samples in the data
    max_len = 0
    for row in list_data:
        max_len = max(max_len, len(row))
    
    # Pad the data to the maximum length
    padded_data = -np.ones((len(list_data), max_len), dtype=np.int32)
    for i, row in enumerate(list_data):
        padded_data[i, :len(row)] = row
    
    return padded_data


def read_int_to_token(vocab_path):
    """ 
    Reads a vocab.json file and returns a dictionary mapping integer to string tokens
    Note: Strings prefixed with Ġ start with a space character

    Args:
        vocab_path (str): path to vocab.json - JSON file mapping tokens to integer intexes

    Returns:
        dict: Maps integer to string tokens
    """

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # get vocab as a reversed dict
    vocab_dict = {v: k for k, v in vocab.items()}

    return vocab_dict


def helper_int_row_to_str(row, vocab_dict):
    decoded_row = ""
    # decoded_data.append([vocab_dict[i] for i in row if i != -1])
    for i in row:
        if i == -1:
            break
        decoded_row += vocab_dict[i]

    # replace Ġ with a space
    decoded_row = decoded_row.replace("Ġ", " ")
    return decoded_row

def decode_bpe_to_text(bpe_data, vocab_dict, n_pools=8):
    """
    Decodes the BPE data into a list of strings without the Ġ characters

    Args:
        bpe_data (numpy.array): (N, M) array of integers
        vocab_dict (dict): Maps integer to string tokens

    Returns:
        list: List of strings
    """
    
    # split the data into chunks of size len(bpe_data) // n_pools
    # chunks = np.array_split(bpe_data, n_pools)
    # zip every chunk with a copy of the vocab_dict
    # inputs = list(zip(chunks, [vocab_dict] * len(chunks)))

    with Pool(n_pools) as p:
        decoded_data = list(p.starmap(helper_int_row_to_str, zip(bpe_data, repeat(vocab_dict))))
        
    # join the list of lists into a list of strings
    # print(type(decoded_data))
    # decoded_data = ["".join(row) for row in decoded_data]
    return decoded_data

