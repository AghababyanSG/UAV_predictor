import os
import csv
import numpy as np
import scipy.io
import pandas as pd
import data_analysis_functions as daf
import matplotlib.pyplot as plt

directory = 'archive/Indoor_signals_1m/6_58G_1m'

files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mat')]

batch_size = 1000
n_batches = len(files) // batch_size + 1

df_mean = pd.DataFrame()

for i in range(n_batches):

    selected_files = files[i * batch_size:(i + 1) * batch_size - 1]
    mats = [scipy.io.loadmat(os.path.join(f)) for f in selected_files]

    df = pd.DataFrame()

    for mat in mats:
        data = mat['sig1']
        single_column = pd.DataFrame(data)
        single_column = single_column.transpose()
        df = pd.concat([df, single_column], axis=1)
    df.columns = range(df.shape[1])
    
    batch_mean = daf.mean_values_to_tuple(df)

    batch_column = pd.DataFrame(batch_mean)
    # batch_column = single_column.transpose()
    df_mean = pd.concat([df_mean, batch_column], axis=1)
    
    # start_idx = i * batch_size
    # end_idx = min((i + 1) * batch_size, len(files))
    
    # batch_files = files[start_idx:end_idx]
    
    # mats = [scipy.io.loadmat(f)['sig1'] for f in batch_files]
    # batch_data = np.concatenate(mats, axis=1)
    
    # df_mean_list[i] = daf.mean_values_to_tuple(pd.DataFrame(batch_data), degree=1)
final_mean = daf.mean_values_to_tuple(df_mean)

with open('mean_values_Indoor_1m_6_all-data.csv', 'w', newline='') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)
    
    # Write the tuple to the CSV file
    csv_writer.writerow(final_mean)

plt.figure(figsize=(100, 50))
plt.plot(final_mean, linestyle='', marker='.')

plt.savefig('mean_values_Indoor_1m_6_all-data.png')
plt.show()
