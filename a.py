import random
import os
import shutil


for i in range(2, 8):
    directory = f'archive/Outdoor_signals_100m/{i}_58G_100m'

    files = [f for f in os.listdir(directory) if f.endswith('.mat')]

    selected_files = random.sample(files, 40)

    destination_directory = f'new_archive/Outdoor_signals_100m/{i}_58G_100m/'

    for file in selected_files:
        shutil.copy(f'archive/Outdoor_signals_100m/{i}_58G_100m' + '/' + file, destination_directory)