import pandas as pd
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
csv_file_path = 'mean_values_Indoor_1m_2_all-data.csv'
csv_data = pd.read_csv(csv_file_path, header=None)
mean_values = torch.tensor(csv_data.values.flatten(), dtype=torch.float).to(device)


file_names = [f'archive/Indoor_signals_1m/3_58G_1m/data{i}.mat' for i in range(1001, 1011)]
errors = []
for file_name in file_names:
    print(file_name)
    
    signal_df = pd.read_csv(csv_file_path, header=None).apply(lambda x: complex(x))
    signal = torch.tensor(signal_df.apply(lambda x: abs(x))).to(device)
    absolute_error = torch.abs(signal - mean_values).to(device)

    mean_absolute_error = torch.mean(absolute_error)
    errors.append(mean_absolute_error.cpu())


df = pd.DataFrame(errors)

# Specify the file path
csv_file_path = 'errors_Indoor_1m_2_all-data.csv'

# Save the DataFrame to the CSV file
df.to_csv(csv_file_path, index=False, header=False)

plt.figure(figsize=(100, 50))
plt.plot(errors, linestyle='', marker='.')

plt.savefig('errors_Indoor_1m_2_all-data.png')
plt.show()
