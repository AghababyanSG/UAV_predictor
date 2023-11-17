import os
import random
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, cuda

def files_into_df(quantity=100, type=2):
    """
    A function to read random files from a certain directory into pd dataframe

    Args:
        quantity (int, optional): The quantity of files. Defaults to 100.
        type (int, optional): Type of UAV. Defaults to 2.

    Returns:
        pandas.DataFrame.dtypes: A dataframe with 1 type UAV signals
    """
    directory = f'archive/Indoor_signals_1m/{type}_58G_1m'

    files = [f for f in os.listdir(directory) if f.endswith('.mat')]

    selected_files = random.sample(files, quantity)
    mats = [scipy.io.loadmat(os.path.join(directory, f)) for f in selected_files]

    df = pd.DataFrame()

    for mat in mats:
        data = mat['sig1']
        single_column = pd.DataFrame(data)
        single_column = single_column.transpose()
        df = pd.concat([df, single_column], axis=1)
    df.columns = range(df.shape[1])
    return df

def mean_values_to_tuple(df, degree=1):
    """Returns mean values in a tuple 

    Args:
        df (pandas.DataFrame.dtypes): Signals of only 1 UAV
        degree (int, optional): Which degree of mean is need to be calculated. Defaults to 1.

    Raises:
        ValueError: Degree can only be 1 or 2

    Returns:
        tuple: Tuple of mean values
    """
    df = df.applymap(lambda x: abs(x))
    df.columns = range(df.shape[1])
    print(df.columns)

    if degree == 1:
        return tuple(df.mean(axis=1))   
    elif degree == 2:
        return tuple(df.mean(axis=1)**2)
    else:
        raise ValueError("Degree can only be 1 or 2.")