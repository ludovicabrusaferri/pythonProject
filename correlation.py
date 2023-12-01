import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def compute_correlation(trueplasma_directory, signaldata_directory):
    trueplasma_files = [file for file in os.listdir(trueplasma_directory) if file.endswith('_TRUEPLASMA.txt')]
    signaldata_files = [file for file in os.listdir(signaldata_directory) if file.endswith('_SIGNALDATA.txt')]

    correlations = []

    # Store Vt values for each subject and ROI
    all_trueplasma_vt_values = []
    all_signaldata_vt_values = []

    for trueplasma_file in trueplasma_files:
        subject_id = trueplasma_file.split('_')[0]
        roi = trueplasma_file.split('_')[1]

        # Find corresponding SIGNALDATA file
        pattern = f'{subject_id}_{roi}'
        matching_signaldata_file = [file for file in signaldata_files if pattern in file]
        matching_trueplasma_file = [file for file in trueplasma_files if pattern in file]

        if len(matching_signaldata_file) > 0:
            # Take the first matching SIGNALDATA file (you may need to adjust this logic)
            signaldata_file = matching_signaldata_file[0]
            trueplasma_file = matching_trueplasma_file[0]

            trueplasma_path = os.path.join(trueplasma_directory, trueplasma_file)
            signaldata_path = os.path.join(signaldata_directory, signaldata_file)

            trueplasma_vt_all = np.loadtxt(trueplasma_path)
            signaldata_vt_all = np.loadtxt(signaldata_path)

            # Append Vt values for the current ROI
            all_trueplasma_vt_values.append(trueplasma_vt_all)
            all_signaldata_vt_values.append(signaldata_vt_all)
    print(all_trueplasma_vt_values)


# Specify the directories for TRUEPLASMA and SIGNALDATA outputs
trueplasma_output_directory = '/Users/luto/Dropbox/AIProject/OUT/TRUEPLASMA/'
signaldata_output_directory = '/Users/luto/Dropbox/AIProject/OUT/SIGNALDATA/'

# Call the function to compute and print the correlation with plotting
average_trueplasma_vt_values = compute_correlation(trueplasma_output_directory, signaldata_output_directory)

# Now you can use average_trueplasma_vt_values for further analysis or plotting
