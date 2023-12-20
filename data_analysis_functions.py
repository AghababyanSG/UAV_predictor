import os
import random
import scipy.io
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import csv


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
    mats = [scipy.io.loadmat(os.path.join(directory, f))
            for f in selected_files]

    df = pd.DataFrame()
    i = 0
    for mat in mats:
        i += 1
        data = mat['sig1']
        single_column = pd.DataFrame(data)
        single_column = single_column.transpose()
        df = pd.concat([df, single_column], axis=1)
    df.columns = range(df.shape[1])
    return df


def file_into_df_column(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    mat = scipy.io.loadtmat(os.path.join(file_path))
    return pd.DataFrame(mat['sig1'])


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

    if degree == 1:
        return tuple(df.mean(axis=1))
    elif degree == 2:
        return tuple(df.mean(axis=1)**2)
    else:
        raise ValueError("Degree can only be 1 or 2.")


def count_files_in_directory(folder_path):
    """Counts how many files are there in a folder

    Args:
        folder_path (str): Path to that folder

    Returns:
        int: number of files in a certain folder
    """
    try:
        file_list = [f for f in os.listdir(
            folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        num_files = len(file_list)
        return num_files
    except FileNotFoundError:
        print(f"Directory not found: {folder_path}")
        return None


def data_to_csv():
    """Saves the needed data into a CSV file
    """

    for type_val in range(2, 5):

        directory = f'archive/Indoor_signals_1m/{type_val}_58G_1m'

        files = [f for f in os.listdir(directory) if f.endswith('.mat')]

        selected_files = random.sample(files, 1000)
        with open(f"trained_files_{type_val}.txt", "w") as output:
            output.write(str(selected_files))
        mats = [scipy.io.loadmat(os.path.join(directory, f))
                for f in selected_files]

        i = 0
        for mat in mats:
            print(i)
            i += 1
            data = mat['sig1']
            # print(type(data))
            # print(data)
            arr = np.array(data[0])
            arr = np.absolute(arr)

            with open('archive/data_1m.csv', 'a', newline='') as csvfile:
                csv.writer(csvfile).writerow(np.append(arr, type_val))
            # single_column = pd.DataFrame(data)
            # single_column = single_column.transpose().abs()
            # single_column.write.mode('append').parquet('archive/data_1m.parquet')
            # print(arr)
            # single_column = pd.DataFrame({str(type_val) + '_' + str(i): arr})
            # single_column_pa = pa.array(arr)
            # column_to_append = pa.Table.from_pandas(single_column)
            # pq.write_table(column_to_append, 'archive/data_1m.parquet')
        print(f'{type_val} is done')


def column_name(path):
    df = pd.read_csv(path)
    cols = len(df.axes[1])
    colum = [f"{i}" for i in range(cols-1)]
    colum.append("type")
    df.columns = colum
    df.to_csv('archive/data_2m.csv')


if __name__ == '__main__':
    data_to_csv()

    # df = df.abs()
    # df.columns = df.columns.astype(str)
    # print(df.head())
    # parquet_file_path = f'archive/dataset.parquet'
    # df.to_parquet(parquet_file_path, engine='fastparquet')

# def data_to_parquet(type=str):
#     path = f'archive/Indoor_signals_1m/{type}_58G_1m'
#     files_count = count_files_in_directory(path)
#     df = files_into_df(quantity=int(files_count * 0.8), type=type)
#     df = df.abs()
#     print(df.head())
#     parquet_file_path = f'archive/Indoor_signals_1m/2_58G_1m/{type}_58G_1m.parquet'
#     df.to_parquet(parquet_file_path, engine='fastparquet')
