import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def convert_csv_to_parquet(input_file_path, output_file_path, drop_option):
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(input_file_path)

    # Remove rows or columns with NaN fields based on the drop_option argument
    if drop_option == 'row':
        df = df.dropna()
    elif drop_option == 'column':
        df = df.dropna(axis=1)

    # Convert Pandas DataFrame to PyArrow Table
    table = pa.Table.from_pandas(df.transpose())

    # Write PyArrow Table to Parquet file
    pq.write_table(table, output_file_path, compression='gzip')

    # Open the Parquet file
    table = pq.read_table(output_file_path)

    # Convert the table to a Pandas DataFrame
    df = table.to_pandas()

    # Print the DataFrame
    print(df.head(100))

# convert_csv_to_parquet('test/data_abs.csv', 'test/data_abs.parquet', 'row')

# import data_analysis_functions as daf

# daf.data_to_csv()

import numpy as np
from csv import writer

# Your NumPy array
new_data = np.array([''])
new_data = np.append(np.append(new_data, np.arange(131072)), 'type')

# Your CSV file
csv_file = "archive/data_1m.csv"

# Read existing data from the CSV file
existing_data = np.genfromtxt(csv_file, delimiter=',', dtype=str)

# Insert the new data at the beginning
all_data = np.insert(existing_data, 0, new_data, axis=0)

# Save the new data to the CSV file
with open('event.csv', 'w', newline='') as f_object:
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerows(all_data)

print("Data appended successfully.")

