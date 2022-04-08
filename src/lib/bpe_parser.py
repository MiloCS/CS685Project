import numpy as np
import os
import json


def read_bpe_data(data_path):
    """
    Reads in the BPE data from the given path into an (N, M) integer numpy array.
    Samples are padded with -1
    N = Number of samples
    M = Maximum number of tokens in an example
    """
    with open(data_path, "r") as f:
        lines = f.readlines()

    def process_row(row):
        # row string " 123 456 789\n" -> integer list [123, 456, 789]
        row = row.strip().split()
        row = list(map(int, row))
        return row

    list_data = list(map(process_row, lines))

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


def decode_bpe_to_text(bpe_data, vocab_dict):
    """
    Decodes the BPE data into a list of strings without the Ġ characters

    Args:
        bpe_data (numpy.array): (N, M) array of integers
        vocab_dict (dict): Maps integer to string tokens

    Returns:
        list: List of strings
    """
    decoded_data = []
    for row in bpe_data:
        decoded_data.append([vocab_dict[i] for i in row if i != -1])
    # join the list of lists into a list of strings
    decoded_data = ["".join(row) for row in decoded_data]
    # remove the Ġ prefixes
    decoded_data = [row.replace("Ġ", " ") for row in decoded_data]
    return decoded_data
