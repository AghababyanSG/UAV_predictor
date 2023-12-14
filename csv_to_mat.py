# Import the libraries
import os
import scipy.io as sio
import pandas as pd

# Define the directory where the CSV files are located
directory = "archive/Indoor_signals_1m/2_58G_1m"

# Loop through the files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file
    if filename.endswith(".csv"):
        # Read the CSV file using pandas
        df = pd.read_csv(os.path.join(directory, filename))
        
        # Convert the DataFrame to a dictionary
        mat_data = {"sig1": df.values}
        
        # Create a MAT file with the same name as the CSV file
        mat_file = filename.replace(".csv", ".mat")
        sio.savemat(os.path.join(directory, mat_file), mat_data)
        
        # Delete the CSV file using os.remove
        os.remove(os.path.join(directory, filename))
        
        # Print a message to indicate the progress
        print(f"Converted {filename} to {mat_file} and deleted {filename}")
    else:
        # Skip the file if it is not a CSV file
        continue
